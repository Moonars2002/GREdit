# GREdit: Geometry-Aware Diffusion and Residual-Guided Densification for 3D Scene Editing

This repository contains the official implementation of GREdit.

## Installation

**Tested on Ubuntu 22.04 + CUDA 11.8 + Python 3.10 (NVIDIA RTX 4090).**
```bash
# Clone the repository
git clone https://github.com/Moonars2002/GREdit.git
cd GREdit

# Create environment
conda create -n gredit python=3.10 -y
conda activate gredit

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## Editing

To edit a 3D scene, you can run the following command.

```bash
python launch.py --config configs/gredit.yaml --train --gpu 0 \
    trainer.max_steps=1500 system.prompt_processor.prompt="YOUR PROMPT" \
    data.source="PATH_TO_DATA" \
    system.gs_source="PATH_TO_PRETRAINED_GS_MODEL"
```
if you find the edited 3DGS has obvious artifacts, you may try to use the original resolution by appending data.use_original_resolution=True to your command. However, doing so will increase GPU memory consumption. To alleviate this dilemma, our proposed Residual-Guided Densification (RGD) effectively mitigates these artifact issues at lower resolutions. It allows you to achieve high-quality, artifact-free editing while keeping the memory footprint highly efficient.

## Evaluation

To calculate the CLIP metric, run the following command：

```bash
python run_clip.py \
  --image_dir0 "/path/to/your/original_images" \
  --image_dir1 "/path/to/your/edited_images" \
  --text0 "source text description" \
  --text1 "target text description" \
  --model "ViT-L/14" \
  --device cuda \
  --batch_size 16 \
  --num_workers 4
```

--interval: The sampling interval for evaluation. Default is 8 (evaluating every 8th image).

--device: Specify cuda for GPU acceleration or cpu.

--batch_size: Batch size for feature extraction. Default is 32.

--num_workers: Number of CPU threads for data loading.

## Acknowledgement
Our implementation is based on the publicly available code from [DGE](https://github.com/silent-chen/DGE)

We also build upon other wonderful repos:

[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

[Threestudio](https://github.com/threestudio-project/threestudio)

[GaussianEditor](https://github.com/buaacyw/GaussianEditor)
