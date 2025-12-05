#!/bin/sh

#SBATCH -J IN-QTAFT
#SBATCH -p qgpu
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH -o slurm_logs/slurm-%j-%a.out
#SBATCH -a 1


root=../data
imagenet_root=../ILSVRC2012
dataset=ImageNet
model_dir="save/qtaft/"

name=QTAFT
image_caption_root=/home/fmg2/waseda/LanguageForVisionRobustness/InternVL/captions/ImageNet/InternVL2_5-8B/Describe_the_image_in_detail_within_50_words.


#################### Hyperparameters ####################
batch_size=128
train_eps=4
train_numsteps=10
train_stepsize=1
epochs=2
total_train_steps=100 # test: only train 100 steps 
lr=1e-5
wd=1e-4
w_i2t=10.0
########################################################

model=clip
arch=vit_b16


python train_qtaft.py \
    --batch_size $batch_size \
    --root $root \
    --imagenet_root $imagenet_root \
    --image_caption_root $image_caption_root \
    --dataset $dataset \
    --model_dir $model_dir \
    --name $name \
    --train_eps $train_eps \
    --train_numsteps $train_numsteps \
    --train_stepsize $train_stepsize \
    --epochs $epochs \
    --learning_rate $lr \
    --weight_decay $wd \
    --arch $arch \
    --model $model \
    --test_eps $train_eps \
    --test_numsteps 10 \
    --w_i2t $w_i2t \
    --total_train_steps $total_train_steps