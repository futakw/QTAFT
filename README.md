# [ACMMM'25] Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models

This repository contains the official implementation of our ACMMM 2025 paper  
**“Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models.”**

![Teaser](assets/teaser.png)
![Method](assets/QT-AFT_method.png)

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