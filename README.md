# [ACMMM'25] Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models

This repository contains the official implementation of our ACMMM 2025 paper  
**“Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models.”**

Author: Futa Waseda

![Teaser](assets/teaser.png)
<br><br>
![Method](assets/QT-AFT-method.png)


## Overview

QTAFT (Quality-Text-guided Adversarial Fine-Tuning) is a simple and effective adversarial fine-tuning framework for vision-language models (VLMs). By leveraging high-quality textual supervision during adversarial training, QTAFT substantially improves adversarial robustness while preserving or even improving clean accuracy.

## Highlights

- 🚀 Improves adversarial robustness of CLIP-based vision-language models
- 📝 Exploits high-quality language supervision during adversarial fine-tuning
- ⚖️ Achieves a better robustness–accuracy trade-off than existing methods
- 🔬 Evaluated on multiple datasets and adversarial attack settings

## Paper

- ACM Digital Library: [(link)](https://dl.acm.org/doi/pdf/10.1145/3746027.3755623)
- arXiv: [(link)](https://arxiv.org/abs/2507.16257)

## Code

This repository contains:
- Training code for QTAFT
- Evaluation scripts
- Pretrained checkpoints
- Reproducible experiment configurations

---

# QTAFT checkpoints CLIP-B/16, CLIP-L/14
- https://huggingface.co/futakw/clip-qt-aft
  
---

# Environment Setup (example using conda)

```bash
conda create -n qtaft python=3.10 -y
conda activate qtaft

conda install pip git -y

pip install numpy==1.26.3 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install ftfy regex tqdm einops scipy
pip install git+https://github.com/fra31/auto-attack 
pip install git+https://github.com/openai/CLIP.git
```

# Data Preparation
## Standard Datasets

Please download datasets (ImageNet-1k, Caltech101, etc.)
following their official instructions.

## ImageNet Captions (required for QTAFT)

Download from:
https://drive.google.com/file/d/1UASPvCz3UiPLSLW_jiajzwYiEYbQsGHb/view?usp=sharing

# Training
- Please modify the dataset paths in `scripts/train_qtaft.sh` before running.
- Example command to run the training script using SLURM:
```
sbatch scripts/train_qtaft.sh
```
- It additionally evaluates the model on the validation set during training.

# Evaluation
- Please modify the dataset paths in `scripts/eval_zeroshot.sh` before running.
- Example command to run the evaluation script using SLURM:
```
sbatch scripts/eval_zeroshot.sh
```


# Citation

If you find this code useful for your research, please consider citing our paper:

```
@inproceedings{waseda2025quality,
  title={Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models},
  author={Waseda, Futa and Sugawara, Saku and Echizen, Isao},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={4808--4816},
  year={2025}
}
```
