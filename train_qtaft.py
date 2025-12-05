# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from dataloader import build_train_loader, build_val_datasets, get_val_loader_list
from eval_utils import build_zeroshot_weights_dict, validate_zeroshot

import clip

from utils.io import Tee, save_checkpoint, fix_model_state_dict
from utils.metrics import accuracy, AverageMeter, ProgressMeter
from utils.lr import cosine_lr
from utils.text_prompts import refine_classname, load_imagenet_folder2name
from utils.model import convert_models_to_fp32

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Device num:", torch.cuda.device_count())

criterion = torch.nn.CrossEntropyLoss().to(device)

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).to(device)
std = torch.tensor(CIFAR100_STD).view(3, 1, 1).to(device)

upper_limit, lower_limit = 1.0, 0.0

IMAGENET_CLASSNAMES_PATH = "data/imagenet_classes_names.txt"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def s2bool(s):
    return s.lower() in ["true", "1", "t", "y", "yes"]

def parse_option():
    parser = argparse.ArgumentParser("Adversarial Finetuning for CLIP")

    # logging / schedule
    parser.add_argument("--print_freq", type=int, default=500, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1000, help="save frequency")
    parser.add_argument("--validate_freq", type=int, default=1, help="validate frequency")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start_step", type=int, default=0, help="start step for this training")
    parser.add_argument(
        "--total_train_steps",
        type=int,
        default=None,
        help="total number of steps for this training",
    )

    # optimization
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--opt_beta1", type=float, default=0.9)
    parser.add_argument("--opt_beta2", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1000, help="warmup steps")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--train_eps", type=float, default=2.0, help="PGD eps (in pixel space)")
    parser.add_argument("--train_numsteps", type=int, default=5)
    parser.add_argument("--train_stepsize", type=float, default=1.0)
    parser.add_argument("--test_eps", type=float, default=2.0, help="eval eps (in pixel space)")
    parser.add_argument("--test_numsteps", type=int, default=5)
    parser.add_argument("--test_stepsize", type=float, default=1.0)
    parser.add_argument("--test_n_samples", type=int, default=1000)

    # model / data
    parser.add_argument("--model", type=str, default="clip", choices=["clip"])
    parser.add_argument("--imagenet_root", type=str, default="../ILSVRC2012")
    parser.add_argument("--arch", type=str, default="vit_b32", choices=["vit_b32", "vit_b16", "vit_l14"])
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--root", type=str, default="../data", help="dataset root")
    parser.add_argument("--dataset", type=str, default="ImageNet", help="train dataset (ImageNet / ImageNet-100)")

    # bookkeeping
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="./save/models")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--filename_suffix", type=str, default="")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--last_num_ft", type=int, default=-1)

    # evaluation config
    parser.add_argument("--out_dir_name", type=str, default="eval_tr")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--template", type=str, default="basic", choices=["basic", "default"])
    parser.add_argument("--zeroshot_weights_dir", type=str, default="templates/zeroshot-weights")
    parser.add_argument("--overwrite_zeroshot_weights", type=s2bool, default=False)
    parser.add_argument("--attack_norm", type=str, default="l_inf", choices=["l_inf", "l_2"])
    parser.add_argument("--CW", action="store_true")
    parser.add_argument("--autoattack", action="store_true")
    parser.add_argument("--reeval_dataset", type=str, nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")

    # caption config
    parser.add_argument("--caps_name", type=str, default="")
    parser.add_argument("--image_caption_root", type=str)

    # training config (caption-only)
    parser.add_argument("--update_text_encoder", default=False, type=s2bool)
    parser.add_argument("--attack", type=str, default="pgd", choices=["pgd"])
    parser.add_argument("--w_i2t", type=float, default=1.0)
    parser.add_argument("--is_update_logit_scale", default=True, type=s2bool)
    parser.add_argument("--attack_bn_mode", type=str, default="eval", choices=["train", "eval"])

    args = parser.parse_args()

    # directory / filename
    args.model_dir = os.path.join(args.model_dir, f"{args.model}_{args.arch}", args.name)
    os.makedirs(args.model_dir, exist_ok=True)
    optim_name = args.optim
    if args.optim == "adamw":
        optim_name = f"adamw-{args.opt_beta1}-{args.opt_beta2}"
    args.filename = "{}_{}_{}_{}_{}_eps{}_lr{}_ep{}_dec{}_b{}_warm{}_loss={}".format(
        args.name,
        args.dataset + "-" + args.caps_name,
        args.model,
        args.arch,
        optim_name,
        args.train_eps,
        args.learning_rate,
        args.epochs,
        args.weight_decay,
        args.batch_size,
        args.warmup,
        args.w_i2t,
    )
    if args.total_train_steps is not None:
        args.filename += f"_totalsteps{args.total_train_steps}"
    args.model_folder = os.path.join(args.model_dir, args.filename)
    os.makedirs(args.model_folder, exist_ok=True)

    return args


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def normalize(X):
    return (X - mu) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode="bicubic")
    X = normalize(X)
    return X


# ---------------------------------------------------------------------------
# CLIP model / data builders
# ---------------------------------------------------------------------------

def build_clip_models(args):
    assert args.model == "clip", "Only CLIP is supported currently."

    arch_map = {
        "vit_b32": "ViT-B/32",
        "vit_b16": "ViT-B/16",
        "vit_l14": "ViT-L/14",
    }
    if args.arch not in arch_map:
        raise ValueError(f"Unknown arch: {args.arch}")

    clip_name = arch_map[args.arch]
    model, _ = clip.load(clip_name, device, jit=False)
    model_orig, _ = clip.load(clip_name, device, jit=False)

    # print parameter num in MB
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    # logit_scale
    if args.is_update_logit_scale:
        model.logit_scale = nn.Parameter(
            torch.ones([], device=device) * np.log(1 / 0.07),
            requires_grad=True,
        )

    # convert to fp32
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()

    convert_models_to_fp32(model_orig)
    model_orig = torch.nn.DataParallel(model_orig)
    model_orig.eval()

    return model, model_orig


def encode_image(model, images):
    return model.module.encode_image(images)


def encode_text(model, texts):
    with torch.no_grad():
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        text_features = model.module.encode_text(text_tokens)
    return text_features

def get_logits(model, image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = model.module.logit_scale.exp()
    logits_per_image = image_features @ text_features.t() * logit_scale
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text


# ---------------------------------------------------------------------------
# Loss (caption-only)
# ---------------------------------------------------------------------------

def l2(out, targets, reduction="none"):
    # squared l2 (no division by dim)
    assert out.shape == targets.shape, f"{out.shape} != {targets.shape}"
    assert out.shape[0] > 1

    squared_error_batch = F.mse_loss(out, targets, reduction="none")
    if reduction == "mean":
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],)
    return squared_error_batch


def compute_caption_loss(
    model,
    img_embed_adv,        # adversarial image embeddings
    img_embed_clean_orig, # clean image embeddings from original CLIP
    txt_embed_caption,    # caption text embeddings
    target_caps,          # caption indices (0..B-1)
    weight_i2t: float,
    reduction="mean",
    return_dict=False,
):
    """
    L = L_unsup (adv vs clean_orig) + w_i2t * CE( img->caption )
    """
    loss_dict = {}

    # unsupervised loss: adv embedding -> original-CLIP's clean embedding 
    unsup_loss = l2(img_embed_adv, img_embed_clean_orig, reduction="mean")
    loss_dict["unsup_loss"] = unsup_loss

    # caption ITC loss (image->caption)
    logits_per_image_caption, _ = get_logits(model, img_embed_adv, txt_embed_caption)
    i2t_loss = criterion(logits_per_image_caption, target_caps)
    loss_dict["caption_i2t_loss"] = i2t_loss * weight_i2t

    loss = sum(loss_dict.values())
    if return_dict:
        return loss, loss_dict
    return loss


class CaptionLossWrapper:
    """
    For PGD: img_embed_adv -> scalar loss
    """
    def __init__(
        self,
        model,
        img_embed_clean_orig,
        txt_embed_caption,
        target_caps,
        weight_i2t,
        reduction="mean",
    ):
        self.model = model
        self.img_embed_clean_orig = img_embed_clean_orig
        self.txt_embed_caption = txt_embed_caption
        self.target_caps = target_caps
        self.weight_i2t = weight_i2t
        self.reduction = reduction

    def __call__(self, img_embed_adv, _targets=None, return_dict=False):
        return compute_caption_loss(
            self.model,
            img_embed_adv=img_embed_adv,
            img_embed_clean_orig=self.img_embed_clean_orig,
            txt_embed_caption=self.txt_embed_caption,
            target_caps=self.target_caps,
            weight_i2t=self.weight_i2t,
            reduction=self.reduction,
            return_dict=return_dict,
        )


# PGD in pixel space (MIM style)
def project_perturbation(perturbation, eps, norm):
    if norm in ["inf", "linf", "Linf"]:
        return torch.clamp(perturbation, -eps, eps)
    elif norm in [2, 2.0, "l2", "L2", "2"]:
        return torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
    else:
        raise NotImplementedError(f"Norm {norm} not supported")


def normalize_grad(grad, p):
    if p in ["inf", "linf", "Linf"]:
        return grad.sign()
    elif p in [2, 2.0, "l2", "L2", "2"]:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)


def pgd(
    forward,
    loss_fn,
    data_clean,
    targets,
    norm,
    eps,
    iterations,
    stepsize,
    output_normalize,
    perturbation=None,
    mode="min",
    momentum=0.9,
    verbose=False,
):
    """
    Minimize or maximize given loss in pixel space (with momentum).
    """
    # ensure in [0,1]
    assert torch.max(data_clean) < 1.0 + 1e-6 and torch.min(data_clean) > -1e-6

    if perturbation is None:
        perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)

    for i in range(iterations):
        perturbation.requires_grad = True
        with torch.enable_grad():
            processed_images = clip_img_preprocessing(data_clean + perturbation)
            try:
                img_embed = forward(processed_images)
            except TypeError:
                img_embed = forward(processed_images)
            if output_normalize:
                img_embed = F.normalize(img_embed, p=2, dim=1)

            loss = loss_fn(img_embed, targets)
            if verbose:
                print(f"[{i}] {loss.item():.5f}")

        with torch.no_grad():
            gradient = torch.autograd.grad(loss, perturbation)[0]
            if gradient.isnan().any():
                print(f"attention: nan in gradient ({gradient.isnan().sum()})")
                gradient[gradient.isnan()] = 0.0

            # normalize
            gradient = normalize_grad(gradient, p=norm)
            # momentum
            velocity = momentum * velocity + gradient
            velocity = normalize_grad(velocity, p=norm)

            # update
            if mode == "min":
                perturbation = perturbation - stepsize * velocity
            elif mode == "max":
                perturbation = perturbation + stepsize * velocity
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # project + clamp to image space
            perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(data_clean + perturbation, 0, 1) - data_clean
            assert not perturbation.isnan().any()
            assert (
                torch.max(data_clean + perturbation) < 1.0 + 1e-6
                and torch.min(data_clean + perturbation) > -1e-6
            )

    return data_clean + perturbation.detach()

# ---------------------------------------------------------------------------
# Train 
# ---------------------------------------------------------------------------

def train(
    train_loader,
    model,
    model_orig,
    optimizer,
    scheduler,
    scaler,
    epoch,
    step,
    args,
):
    global best_acc1

    model_orig.eval()

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix=f"Epoch: [{epoch}]",
    )

    # train mode for vision encoder
    model.module.visual.train()

    end = time.time()
    for i, (images, target_labels, captions, idx) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        batch_size = images.size(0)
        if batch_size != args.batch_size:
            print("Ignore the last batch for simplicity.")
            continue

        # scheduler step
        step += 1
        if step % args.print_freq == 0:
            print("Step:", step)
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target_labels = target_labels.to(device)
        idx = idx.to(device)

        # captions
        txt_embed_caption = encode_text(model, captions)
        target_caps = torch.arange(batch_size, device=device)

        with autocast():
            # original embedding (Forig(x))
            images_clean = clip_img_preprocessing(images)
            with torch.no_grad():
                img_embed_clean_orig = encode_image(model_orig, images_clean)

            # ---- adversarial attack (PGD) ----
            if args.attack_bn_mode == "eval":
                model.eval()
            else:
                model.train()
            assert args.attack == "pgd"

            loss_fn = CaptionLossWrapper(
                model=model,
                img_embed_clean_orig=img_embed_clean_orig,
                txt_embed_caption=txt_embed_caption,
                target_caps=target_caps,
                weight_i2t=args.w_i2t,
                reduction="mean",
            )

            data_adv = pgd(
                forward=model.module.encode_image,
                loss_fn=loss_fn,
                data_clean=images,
                targets=target_caps,
                norm="linf",
                eps=args.train_eps,
                iterations=args.train_numsteps,
                stepsize=args.train_stepsize,
                output_normalize=False,
                perturbation=torch.zeros_like(images)
                .uniform_(-args.train_eps, args.train_eps)
                .requires_grad_(True),
                mode="max",
                verbose=False,
            )
            images_adv = clip_img_preprocessing(data_adv)

            # --- compute loss ---
            model.train()
            img_embed_adv = encode_image(model, images_adv)

            loss, loss_dict = compute_caption_loss(
                model=model,
                img_embed_adv=img_embed_adv,
                img_embed_clean_orig=img_embed_clean_orig,
                txt_embed_caption=txt_embed_caption,
                target_caps=target_caps,
                weight_i2t=args.w_i2t,
                reduction="mean",
                return_dict=True,
            )

            # ----- For logging (caption retrieval accuracy) -----
            with torch.no_grad():
                img_embed_norm = img_embed_adv / img_embed_adv.norm(dim=-1, keepdim=True)
                txt_embed_norm = txt_embed_caption / txt_embed_caption.norm(dim=-1, keepdim=True)
                output = img_embed_norm @ txt_embed_norm.t()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # clamp logit_scale (ln(100))
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy (caption index)
        acc1 = accuracy(output, target_caps, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.is_update_logit_scale:
                print("logit_scale:", model.module.logit_scale.exp())
            print("loss_dict:", loss_dict)

        # periodic checkpoint
        if step % args.save_freq == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "step": step,
                    "vision_encoder_state_dict": model.module.visual.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                args,
                filename=f"checkpoint_{step}.pth.tar",
            )
            # remove previous checkpoints (to save space)
            if (step - args.save_freq) % 5000 != 0:
                prev_ckpt_path = os.path.join(
                    args.model_folder, f"checkpoint_{step - args.save_freq}.pth.tar"
                )
                if os.path.exists(prev_ckpt_path):
                    os.remove(prev_ckpt_path)

        if args.total_train_steps is not None and step >= args.total_train_steps:
            print("Training finished at step:", step)
            break

        if args.debug and i > 10:
            print("Debug mode. Breaking...")
            break

    return losses.avg, top1.avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global best_acc1, device, args

    # name for eval results
    res_name = "results_eps{}_numsteps{}_stepsize{}".format(
        args.test_eps, args.test_numsteps, args.test_stepsize
    )
    if args.autoattack:
        res_name += "_autoattack"
    elif args.CW:
        res_name += "_CW"
    else:
        res_name += "_PGD"
    if args.test_n_samples < 1000000:
        res_name += f"_n_samples{args.test_n_samples}"
    print("\nres_name:", res_name)

    # keep original integer eps/stepsize for logging / post-eval
    train_eps_int = args.train_eps
    train_stepsize_int = args.train_stepsize

    # convert to [0,1] scale
    args.train_eps = args.train_eps / 255.0
    args.test_eps = args.test_eps / 255.0
    args.train_stepsize = args.train_stepsize / 255.0
    args.test_stepsize = args.test_stepsize / 255.0

    # redirect stdout / stderr AFTER basic prints if you like them in console
    sys.stdout = Tee(os.path.join(args.model_folder, "out.txt"))
    sys.stderr = Tee(os.path.join(args.model_folder, "err.txt"))

    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    imagenet_root = args.imagenet_root

    # ------------ Model -----------
    model, model_orig = build_clip_models(args)

    # ------------- optimizer -------------
    if args.update_text_encoder:
        params = list(model.module.parameters())
        print("Update both vision and text encoder!")
    else:
        if args.last_num_ft == -1:
            params = list(model.module.visual.parameters())
        else:
            params = list(model.module.visual.parameters())[-args.last_num_ft :]
        print("Update only vision encoder!")

    if args.is_update_logit_scale:
        print("Update logit_scale")
        params += [model.module.logit_scale]

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.opt_beta1, args.opt_beta2),
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    # -------------- Data Loaders --------------
    preprocess224 = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    # train dataset
    train_dataset, train_loader = build_train_loader(args, imagenet_root, preprocess224)

    # val dataset
    val_dataset_name, val_dataset_list, classnames_dict = build_val_datasets(args)
    val_loader_list = get_val_loader_list(args, val_dataset_list)
    zeroshot_weights_dict = build_zeroshot_weights_dict(
        model, device, val_dataset_name, classnames_dict, args
    )
    criterion_eval = nn.CrossEntropyLoss().to(device)

    # -------------- Training / Evaluation --------------
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    results_dict_each_epoch = {}

    args.start_epoch = 0
    if args.total_train_steps is None:
        args.total_train_steps = len(train_loader) * args.epochs

    # optional: evaluate only
    if args.evaluate:
        acc1_mean, results_dict = validate_zeroshot(
            val_loader_list,
            val_dataset_name,
            zeroshot_weights_dict,
            model,
            criterion_eval,
            args,
            max_num=args.test_n_samples, 
            out_dir=args.model_folder,  # or separate eval dir
            save_name=res_name,
        )
        results_dict_each_epoch["eval"] = results_dict
        print("Eval-only finished. Acc:", acc1_mean)
        return

    epochs_since_improvement = 0

    for epoch in range(args.start_epoch, args.epochs):
        step = (epoch - args.start_epoch) * len(train_loader) + args.start_step

        # train for one epoch
        train(
            train_loader,
            model,
            model_orig,
            optimizer,
            scheduler,
            scaler,
            epoch,
            step,
            args,
        )

        # save last checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "step": step,
                "vision_encoder_state_dict": model.module.visual.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            args,
            filename="checkpoint_last.pth.tar",
        )

        # eval on a subset of datasets
        if epoch % args.validate_freq == 0:
            acc1_mean, results_dict = validate_zeroshot(
                val_loader_list[:3],
                val_dataset_name[:3],
                zeroshot_weights_dict,
                model,
                criterion_eval,
                args,
                max_num=args.test_n_samples, 
                out_dir=args.model_folder,  # or separate eval dir
                save_name=None,
            )
            results_dict_each_epoch[epoch] = results_dict

    print("Experiment finished.")

    # save results
    res_path = os.path.join(args.model_folder, res_name + "_dict.json")
    with open(res_path, "w") as f:
        json.dump(results_dict_each_epoch, f, indent=4)

    # mark done
    with open(os.path.join(args.model_folder, "done"), "w") as f:
        f.write("done")

    # Evaluation after training
    if not args.evaluate:
        save_dir = os.path.join(args.model_folder, args.out_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        # PGD-10
        pgd10_args = copy.deepcopy(args)
        pgd10_args.test_numsteps = 10
        res_name_pgd10 = "results_eps{}_numsteps{}_stepsize{}".format(
            train_eps_int, pgd10_args.test_numsteps, train_stepsize_int
        )
        res_name_pgd10 += "_PGD"
        assert not args.autoattack and not args.CW
        print("\nres_name:", res_name_pgd10)
        acc1_mean, results_dict = validate_zeroshot(
            val_loader_list,
            val_dataset_name,
            zeroshot_weights_dict,
            model,
            criterion_eval,
            pgd10_args,
            max_num=pgd10_args.test_n_samples, 
            out_dir=save_dir, 
            save_name=res_name_pgd10,
        )
        print("PGD-10:", acc1_mean)

        # AutoAttack
        auto_args = copy.deepcopy(args)
        auto_args.test_numsteps = 100
        auto_args.autoattack = True
        auto_args.test_n_samples = 1000
        res_name_auto = "results_eps{}_numsteps{}_stepsize{}".format(
            train_eps_int, auto_args.test_numsteps, train_stepsize_int
        )
        res_name_auto += f"_n_samples{auto_args.test_n_samples}_autoattack"
        print("\nres_name:", res_name_auto)
        acc1_mean, results_dict = validate_zeroshot(
            val_loader_list,
            val_dataset_name,
            zeroshot_weights_dict,
            model,
            criterion_eval,
            auto_args,
            max_num=auto_args.test_n_samples, 
            out_dir=save_dir, 
            save_name=res_name_auto,
        )
        print("AutoAttack:", acc1_mean)


if __name__ == "__main__":
    args = parse_option()
    # save args once here (after parse)
    with open(os.path.join(args.model_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    main()