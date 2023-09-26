#!/bin/bash

# A bash script that runs some of the `run` commands in the cog.yaml
# done to quickly get a prototype up and running - boot times would include downloading otherwise

# Clone the repo if it doesn't exist
REPO_DIR="models/StableDiffusion/stable-diffusion-v1-5"
if [ ! -d "$REPO_DIR" ]; then
    git clone --branch fp16 https://huggingface.co/runwayml/stable-diffusion-v1-5 $REPO_DIR
else
    echo "Repository already exists at $REPO_DIR, skipping clone."
fi

# Ensure directories exist
mkdir -p models/Motion_Module
mkdir -p models/MotionLoRA
mkdir -p models/DreamBooth_LoRA

# Download Motion_Module models
wget -O models/Motion_Module/mm_sd_v14.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt
wget -O models/Motion_Module/mm_sd_v15.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt
wget -O models/Motion_Module/mm_sd_v15_v2.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt

# Download MotionLoRA models
wget -O models/MotionLoRA/v2_lora_ZoomIn.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt
wget -O models/MotionLoRA/v2_lora_ZoomOut.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt
wget -O models/MotionLoRA/v2_lora_PanLeft.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt
wget -O models/MotionLoRA/v2_lora_PanRight.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt
wget -O models/MotionLoRA/v2_lora_PanUp.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanUp.ckpt
wget -O models/MotionLoRA/v2_lora_PanDown.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanDown.ckpt
wget -O models/MotionLoRA/v2_lora_RollingClockwise.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt
wget -O models/MotionLoRA/v2_lora_RollingAnticlockwise.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt

# Download DreamBooth_LoRA models
wget -O models/DreamBooth_LoRA/toonyou_beta3.safetensors https://civitai.com/api/download/models/78775
wget -O models/DreamBooth_LoRA/lyriel_v16.safetensors https://civitai.com/api/download/models/72396
wget -O models/DreamBooth_LoRA/rcnzCartoon3d_v10.safetensors https://civitai.com/api/download/models/71009
wget -O models/DreamBooth_LoRA/majicmixRealistic_v5Preview.safetensors https://civitai.com/api/download/models/79068
# wget -O models/DreamBooth_LoRA/realisticVisionV40_v20Novae.safetensors https://civitai.com/api/download/models/29460
wget -O models/DreamBooth_LoRA/realisticVisionV20_v20.safetensors https://huggingface.co/ckpt/realistic-vision-v20/resolve/main/realisticVisionV20_v20.safetensors


# need models/DreamBooth_LoRA/realisticVisionV20_v20.safetensors ?