# attacks.py
import torch
import torch.nn.functional as F
import numpy as np
from autoattack import AutoAttack
from autoattack.other_utils import L2_norm
from utils.text_prompts import one_hot_embedding 
from utils.model import convert_models_to_fp32
import functools
from torch.cuda.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(device)
std = torch.tensor(CIFAR100_STD).view(3,1,1).to(device)

upper_limit, lower_limit = 1, 0

def normalize(X):
    return (X - mu) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode="bicubic")
    X = normalize(X)
    return X

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def forward_classification(images_unnorm, model, zeroshot_weights):
    # 旧 eval と完全同じロジック
    images = clip_img_preprocessing(images_unnorm)
    if hasattr(model, "module"):
        image_features = model.module.encode_image(images)
    else:
        image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ zeroshot_weights
    return logits


# Randomized defense check
from autoattack.other_utils import L2_norm

checks_doc_path = "docs/checks.md"


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.0
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
            # print(diff.max().item(), max_diff)
    if any(acc) or max_diff > alpha:
        msg = (
            'it seems to be a randomized defense! Please use version="rand".'
            + f" See {checks_doc_path} for details."
        )
        return True
    return False


def attack_auto(
    model,
    images,
    target,
    zeroshot_weights,
    attacks_to_run=["apgd-ce", "apgd-dlr"],
    epsilon=0,
):
    convert_models_to_fp32(model)
    model.eval()
    forward_pass = functools.partial(
        forward_classification,
        model=model,
        zeroshot_weights=zeroshot_weights,
    )

    # adversary = AutoAttack(forward_pass, norm="Linf", eps=epsilon, version="standard", verbose=False)
    # adversary.attacks_to_run = attacks_to_run
    # x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    # return x_adv
    is_random = check_randomized(
        forward_pass, images, target, bs=images.shape[0], n=5, alpha=1e-4, logger=None
    )
    if is_random:
        with autocast(False):
            adversary = AutoAttack(
                forward_pass, norm="Linf", eps=epsilon, version="standard", verbose=False
            )
            adversary.attacks_to_run = attacks_to_run
            x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    else:
        adversary = AutoAttack(forward_pass, norm="Linf", eps=epsilon, version="standard", verbose=False)
        adversary.attacks_to_run = attacks_to_run
        x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv


def attack_pgd(
    model,
    criterion,
    X,
    target,
    zeroshot_weights,
    alpha,
    attack_iters,
    norm,
    restarts=1,
    early_stop=True,
    epsilon=0,
    text_features=None,
):
    delta = torch.zeros_like(X).to(device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        images = X + delta
        output = forward_classification(images, model, zeroshot_weights)
        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_CW(
    model,
    criterion,
    X,
    target,
    alpha,
    zeroshot_weights,
    attack_iters,
    norm,
    restarts=1,
    early_stop=True,
    epsilon=0,
):
    delta = torch.zeros_like(X).to(device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        images = X + delta
        output = forward_classification(images, model, zeroshot_weights)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.to(device)

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = -torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta
