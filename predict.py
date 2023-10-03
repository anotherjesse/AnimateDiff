# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/gen/docs/python.md

import hashlib
import imageio
from typing import List
from cog import BasePredictor, Input
from cog import Path as CogPath

import datetime
import inspect
import os
from omegaconf import OmegaConf
import torchvision
from einops import rearrange
import numpy as np

import torch

from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available


from dataclasses import dataclass


@dataclass
class Arguments:
    pretrained_model_path: str = "models/StableDiffusion/stable-diffusion-v1-5"
    inference_config: str = "configs/inference/inference-v1.yaml"
    config: str = ""
    L: int = 16
    W: int = 512
    H: int = 512


FAKE_YAML_TEMPLATE = """
Cog:
  inference_config: "{inference_config}"
  motion_module:
    - "{motion_module}"
  {motion_module_lora_configs_section}
  lora_model_path: "{lora_model_path}"
  seed:           {seed}
  steps:          {steps}
  guidance_scale: {guidance_scale}
  prompt:
    - "{prompt}"
  n_prompt:
    - "{negative_prompt}"
"""

MOTION_MODULE_LORA_CONFIG_TEMPLATE = """
    - path:  "models/MotionLoRA/v2_lora_{motion_lora_type}.ckpt"
      alpha: {motion_lora_strength}
"""


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.real = os.path.exists("/src/weights")

        if not self.real:
            print("This is a template model - add diffusers weights to /src/weights")
            return
        self.inference_config_path = "configs/inference/inference-v2.yaml"
        self.pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
        self.pretrained_model_path = "/src/weights"

        self.inference_config = OmegaConf.load(self.inference_config_path)

        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_path, subfolder="vae")
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            self.pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs),
        )

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            assert False

        self.pipeline = AnimationPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

    def custom_save_videos_grid(self, videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
        videos = rearrange(videos, "b c t h w -> t b c h w")
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save as .mp4
        writer = imageio.get_writer(path, fps=fps)
        for frame in outputs:
            writer.append_data(frame)
        writer.close()

    def gen(self, args):
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        # Compute a hash of the config string to get a fixed length string.
        config_hash = hashlib.md5(args.config.encode()).hexdigest()

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{config_hash}-{time_str}"
        os.makedirs(savedir)

        if os.path.exists(args.config):  # Check if args.config is a file path
            config = OmegaConf.load(args.config)
        else:  # If not, then parse it as a raw YAML string
            config = OmegaConf.create(args.config)
        samples = []

        sample_idx = 0
        for model_idx, (config_key, model_config) in enumerate(list(config.items())):
            motion_modules = model_config.motion_module
            motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
            for motion_module in motion_modules:
                # Check if the current hash is different from the previous one
                current_hash = hashlib.md5((str(motion_module) + str(model_config)).encode()).hexdigest()
                if not hasattr(self, "previous_hash") or self.previous_hash != current_hash:
                    self.pipeline = load_weights(
                        self.pipeline,
                        # motion module
                        motion_module_path=motion_module,
                        motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
                        # image layers
                        # dreambooth_model_path=model_config.get("dreambooth_path", ""),
                        lora_model_path=model_config.get("lora_model_path", ""),
                        lora_alpha=model_config.get("lora_alpha", 0.8),
                    ).to("cuda")
                    # Update the previous hash to the current one
                    self.previous_hash = current_hash

                prompts = model_config.prompt
                n_prompts = (
                    list(model_config.n_prompt) * len(prompts)
                    if len(model_config.n_prompt) == 1
                    else model_config.n_prompt
                )

                random_seeds = model_config.get("seed", [-1])
                random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
                random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

                config[config_key].random_seed = []
                for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(
                    zip(prompts, n_prompts, random_seeds)
                ):
                    # manually set random seed for reproduction
                    if random_seed != -1:
                        torch.manual_seed(random_seed)
                    else:
                        torch.seed()
                    config[config_key].random_seed.append(torch.initial_seed())

                    print(f"current seed: {torch.initial_seed()}")
                    print(f"sampling {prompt} ...")
                    sample = self.pipeline(
                        prompt,
                        negative_prompt=n_prompt,
                        num_inference_steps=model_config.steps,
                        guidance_scale=model_config.guidance_scale,
                        width=args.W,
                        height=args.H,
                        video_length=args.L,
                    ).videos
                    samples.append(sample)

                    prompt = "".join(prompt.split(" ")[:5])
                    prompt = "".join(e for e in prompt if e.isalnum())

                    os.makedirs(os.path.dirname(f"{savedir}/sample/"), exist_ok=True)
                    self.custom_save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.mp4")
                    yield CogPath(f"{savedir}/sample/{sample_idx}-{prompt}.mp4")
                    print(f"save to {savedir}/sample/{prompt}.mp4")

                    sample_idx += 1

        samples = torch.concat(samples)
        save_videos_grid(samples, f"{savedir}/sample.mp4", n_rows=4)

        OmegaConf.save(config, f"{savedir}/config.yaml")

    def set_defaults_for_ints_and_floats(
        self,
        zoom_in_motion_strength,
        zoom_out_motion_strength,
        pan_left_motion_strength,
        pan_right_motion_strength,
        pan_up_motion_strength,
        pan_down_motion_strength,
        rolling_clockwise_motion_strength,
        rolling_anticlockwise_motion_strength,
        steps,
        guidance_scale,
        frames,
        width,
        height,
        seed,
    ):
        zoom_in_motion_strength = zoom_in_motion_strength if zoom_in_motion_strength is not None else 0.0
        zoom_out_motion_strength = zoom_out_motion_strength if zoom_out_motion_strength is not None else 0.0
        pan_left_motion_strength = pan_left_motion_strength if pan_left_motion_strength is not None else 0.0
        pan_right_motion_strength = pan_right_motion_strength if pan_right_motion_strength is not None else 0.0
        pan_up_motion_strength = pan_up_motion_strength if pan_up_motion_strength is not None else 0.0
        pan_down_motion_strength = pan_down_motion_strength if pan_down_motion_strength is not None else 0.0
        rolling_clockwise_motion_strength = (
            rolling_clockwise_motion_strength if rolling_clockwise_motion_strength is not None else 0.0
        )
        rolling_anticlockwise_motion_strength = (
            rolling_anticlockwise_motion_strength if rolling_anticlockwise_motion_strength is not None else 0.0
        )
        steps = steps if steps is not None else 25
        guidance_scale = guidance_scale if guidance_scale is not None else 7.5
        frames = frames if frames is not None else 16
        width = width if width is not None else 512
        height = height if height is not None else 512
        if seed is None or seed <= 0:
            seed = -1

        return (
            zoom_in_motion_strength,
            zoom_out_motion_strength,
            pan_left_motion_strength,
            pan_right_motion_strength,
            pan_up_motion_strength,
            pan_down_motion_strength,
            rolling_clockwise_motion_strength,
            rolling_anticlockwise_motion_strength,
            steps,
            guidance_scale,
            frames,
            width,
            height,
            seed,
        )

    def predict(
        self,
        prompt: str = Input(
            default="photo of vocano, rocks, storm weather, wind, lava waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        ),
        negative_prompt: str = Input(
            default="blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        ),
        steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=100,
            default=25,
        ),
        guidance_scale: float = Input(
            description="Guidance Scale. How closely do we want to adhere to the prompt and its contents",
            ge=0.0,
            le=20,
            default=7.5,
        ),
        frames: int = Input(
            description="Length of the video in frames (playback is at 8 fps e.g. 16 frames @ 8 fps is 2 seconds)",
            default=16,
            ge=1,
            le=32,
        ),
        width: int = Input(
            description="Width in pixels",
            default=512,
        ),
        height: int = Input(
            description="Height in pixels",
            default=512,
        ),
        seed: int = Input(
            description="Seed for different images and reproducibility. Use -1 to randomise seed",
            default=-1,
        ),
        zoom_in_motion_strength: float = Input(
            description="Strength of Zoom In Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        zoom_out_motion_strength: float = Input(
            description="Strength of Zoom Out Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        pan_left_motion_strength: float = Input(
            description="Strength of Pan Left Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        pan_right_motion_strength: float = Input(
            description="Strength of Pan Right Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        pan_up_motion_strength: float = Input(
            description="Strength of Pan Up Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        pan_down_motion_strength: float = Input(
            description="Strength of Pan Down Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        rolling_clockwise_motion_strength: float = Input(
            description="Strength of Rolling Clockwise Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        rolling_anticlockwise_motion_strength: float = Input(
            description="Strength of Rolling Anticlockwise Motion LoRA. 0 disables the LoRA",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
    ) -> List[CogPath]:
        if not self.real:
            raise Exception("This is a template model, for use in dreambooth")

        (
            zoom_in_motion_strength,
            zoom_out_motion_strength,
            pan_left_motion_strength,
            pan_right_motion_strength,
            pan_up_motion_strength,
            pan_down_motion_strength,
            rolling_clockwise_motion_strength,
            rolling_anticlockwise_motion_strength,
            steps,
            guidance_scale,
            frames,
            width,
            height,
            seed,
        ) = self.set_defaults_for_ints_and_floats(
            zoom_in_motion_strength,
            zoom_out_motion_strength,
            pan_left_motion_strength,
            pan_right_motion_strength,
            pan_up_motion_strength,
            pan_down_motion_strength,
            rolling_clockwise_motion_strength,
            rolling_anticlockwise_motion_strength,
            steps,
            guidance_scale,
            frames,
            width,
            height,
            seed,
        )

        lora_model_path = ""
        motion_module_type = "mm_sd_v15_v2"
        pretrained_model_path = self.pretrained_model_path
        inference_config_path = self.inference_config_path
        motion_module = f"models/Motion_Module/{motion_module_type}.ckpt"
        # dreambooth_path = f"/src/weights"
        motion_strengths = {
            "ZoomIn": zoom_in_motion_strength,
            "ZoomOut": zoom_out_motion_strength,
            "PanLeft": pan_left_motion_strength,
            "PanRight": pan_right_motion_strength,
            "PanUp": pan_up_motion_strength,
            "PanDown": pan_down_motion_strength,
            "RollingAnticlockwise": rolling_anticlockwise_motion_strength,
            "RollingClockwise": rolling_clockwise_motion_strength,
        }

        motion_module_lora_configs = ""
        for motion_lora_type, motion_lora_strength in motion_strengths.items():
            if motion_lora_strength != 0:
                motion_module_lora_configs += MOTION_MODULE_LORA_CONFIG_TEMPLATE.format(
                    motion_lora_type=motion_lora_type, motion_lora_strength=motion_lora_strength
                )

        motion_module_lora_configs_section = (
            f"motion_module_lora_configs:{motion_module_lora_configs}" if motion_module_lora_configs else ""
        )

        # Replace placeholders directly in the template
        config = FAKE_YAML_TEMPLATE.format(
            inference_config=inference_config_path,
            motion_module=motion_module,
            motion_module_lora_configs_section=motion_module_lora_configs_section,
            # dreambooth_path=dreambooth_path,
            lora_model_path=lora_model_path,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        print(f"{'-'*80}")
        print(config)
        print(f"{'-'*80}")

        args = Arguments(
            pretrained_model_path=pretrained_model_path,
            inference_config=inference_config_path,
            config=config,
            L=frames,
            W=width,
            H=height,
        )

        yield from self.gen(args)
