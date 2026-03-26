GREdit: Geometry-Aware Diffusion and Residual-Guided Densification for 3D Scene Editing
This repository contains the official implementation of GREdit. We propose a novel 3D scene editing framework based on 3D Gaussian Splatting (3DGS) that introduces Geometry-Aware Diffusion Guidance (GDG) and Residual-Guided Densification (RGD) to achieve precise and consistent edits while preserving structural integrity.

⚙️ INSTALLATION
Tested on Ubuntu 22.04 + CUDA 11.8 + Python 3.10 (NVIDIA RTX 4090).

1. Clone our repo:

Bash
git clone https://github.com/Moonars2002/GREdit.git
cd GREdit

2. Create conda environment:

Bash
conda create -n gredit python=3.10
conda activate gredit

3. Install PyTorch:

Bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Bash
pip install -r requirements.txt
