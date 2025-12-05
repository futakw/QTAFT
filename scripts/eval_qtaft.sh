#!/bin/sh
    
#SBATCH -J eval-zs
#SBATCH -p qgpu-debug
#SBATCH --gres=gpu:tesla_a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH -o slurm_logs/slurm-%j-%a.out
#SBATCH -a 1


root=../data
imagenet_root=../ILSVRC2012
batch_size=16
test_eps=4
arch=vit_b16
out_dir_name="eval_zeroshot"
path="./save/qtaft/ImageNet-caps-/clip-vit_b16/QTAFT/QTAFT_ImageNet_clip_vit_b16_adamw-0.9-0.95_eps4.0_lr1e-05_ep2_dec0.0001_b128_warm1000_loss=10.0/checkpoint_10000.pth.tar"
model=clip
template=basic
overwrite_zeroshot_weights=f


        
# APGD-100, 1000 images
python eval_zeroshot.py \
    --model $model \
    --arch $arch \
    --load_path $path \
    --test_batch_size $batch_size \
    --root $root \
    --imagenet_root $imagenet_root \
    --test_eps $test_eps \
    --test_numsteps 100 \
    --test_n_samples 1000 \
    --out_dir_name $out_dir_name \
    --template $template \
    --overwrite_zeroshot_weights $overwrite_zeroshot_weights \
    --autoattack