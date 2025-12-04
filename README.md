# [ACMMM'25] Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models

# Environment Setup (example using conda)
```
conda create -n qtaft python=3.10 -y
conda activate qtaft

conda install pip git -y

pip install numpy==1.26.3 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install ftfy regex tqdm einops
pip install git+https://github.com/fra31/auto-attack 
pip install git+https://github.com/openai/CLIP.git
```

# Data Preparation
- Datasets: please download the datasets (e.g., ImageNet, CIFAR100) following the instructions on their official websites.
- ImageNet Captions: please download the ImageNet captions from [here](https://github

# Training
- Please modify the dataset paths in `scripts/train_qtaft.sh` before running.
- Example command to run the training script using SLURM:
```
sbatch scripts/train_qtaft.sh
```