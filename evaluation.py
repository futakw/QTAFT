# evaluation.py
import os
import json
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import numpy as np
from attacks import (
    forward_classification,
    attack_auto,
    attack_pgd,
    attack_CW,
)
from utils.metrics import accuracy, AverageMeter, ProgressMeter
from torch.cuda.amp import autocast
import torch.nn as nn
import clip

def zero_shot_classifier(model, classnames, templates, device, amp=True, model_type="clip"):
    autocast_ctx = torch.cuda.amp.autocast if amp else torch.no_grad
    zeroshot_weights = []
    with torch.no_grad(), autocast_ctx():
        for classname in tqdm(classnames):
            texts = [temp.format(c=classname) for temp in templates]

            texts_token = clip.tokenize(texts, truncate=True).to(device)
            text_features = model.module.encode_text(texts_token)

            class_embedding = F.normalize(text_features, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def build_templates_dict(val_dataset_name, template_mode):
    if template_mode == "basic":
        templates = ["This is a photo of a {c}"]
        return {name: templates for name in val_dataset_name}
    elif template_mode == "default":
        with open("data/zeroshot-templates.json", "r") as f:
            all_templates = json.load(f)
        val_dataname2key = {
            "cifar10": "cifar10",
            "cifar100": "cifar100",
            "STL10": "stl10",
            "SUN397": "sun397",
            "StanfordCars": "cars",
            "Food101": "food101",
            "oxfordpet": "pets",
            "flowers102": "flowers",
            "dtd": "dtd",
            "EuroSAT": "eurosat",
            "fgvc_aircraft": "fgvc_aircraft",
            "PCAM": "pcam",
            "ImageNet": "imagenet1k",
            "Caltech101": "caltech101",
            "ImageNet-S": "imagenet1k",
            "ImageNet-R": "imagenet1k",
            "ImageNet-v2": "imagenet1k",
            "ImageNet-O": "imagenet1k",
            "ImageNet-A": "imagenet1k",
        }
        return {name: all_templates[val_dataname2key[name]] for name in val_dataset_name}
    else:
        raise ValueError(f"Unknown template mode: {template_mode}")


def build_zeroshot_weights_dict(
    model,
    device,
    val_dataset_name,
    classnames_dict,
    args,
):
    # TEMPLATE_LIST は args.template のみでOKなら１つ
    TEMPLATE = args.template
    ZEROSHOT_WEIGHTS_DIR = os.path.join(args.zeroshot_weights_dir, args.arch, TEMPLATE)
    os.makedirs(ZEROSHOT_WEIGHTS_DIR, exist_ok=True)

    # templates_dict を旧 eval と同じように構築
    templates_dict = build_templates_dict(val_dataset_name, TEMPLATE)

    zeroshot_weights_dict = {}
    for dataset_name in val_dataset_name:
        Z_PATH = os.path.join(ZEROSHOT_WEIGHTS_DIR, f"{dataset_name}.pt")
        if os.path.exists(Z_PATH) and not args.overwrite_zeroshot_weights:
            zeroshot_weights = torch.load(Z_PATH, map_location=device)
        else:
            if dataset_name not in templates_dict:
                print(f"Templates for {dataset_name} not found. Skipping...")
                continue
            classnames = classnames_dict[dataset_name]
            templates = templates_dict[dataset_name]
            zeroshot_weights = zero_shot_classifier(
                model, classnames, templates, device, amp=True, model_type=args.model
            )
            torch.save(zeroshot_weights, Z_PATH)
        zeroshot_weights_dict[dataset_name] = zeroshot_weights
    return zeroshot_weights_dict


def validate_zeroshot(
    val_loader_list,
    val_dataset_name,
    zeroshot_weights_dict,
    model,
    criterion,
    args,
    max_num=np.inf,
    device="cuda",
    out_dir=None,
    save_name=None,
):
    dataset_num = len(val_loader_list)
    acc_all = []
    results_dict = {}

    if out_dir is not None and save_name is not None:
        res_path = os.path.join(out_dir, f"{save_name}.json")
        if os.path.exists(res_path):
            print(f"Results already exist at {res_path}. Loading...")
            with open(res_path, "r") as f:
                results_dict = json.load(f)

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):
        val_loader = val_loader_list[cnt]
        dataset_name = val_dataset_name[cnt]

        if dataset_name not in zeroshot_weights_dict:
            print(f"Zeroshot weights for {dataset_name} not found. Skipping...")
            continue
        zeroshot_weights = zeroshot_weights_dict[dataset_name]

        if dataset_name in args.reeval_dataset:
            print(f"Re-evaluating {dataset_name}...")
        elif dataset_name in results_dict and not args.overwrite:
            print(f"Results for {dataset_name} already exist. Skipping...")
            acc_all.append(results_dict[dataset_name]["top1_adv_prompt"])
            continue

        binary = ["PCAM", "hateful_memes"]
        attacks_to_run = ["apgd-ce", "apgd-dlr"]
        if dataset_name in binary:
            attacks_to_run = ["apgd-ce"]

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1_org = AverageMeter("Original Acc@1", ":6.2f")
        top1_prompt = AverageMeter("AT-Model Acc@1", ":6.2f")
        top1_adv_org = AverageMeter("Adv Original Acc@1", ":6.2f")
        top1_adv_prompt = AverageMeter("Adv AT-Model Acc@1", ":6.2f")

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
            prefix=dataset_name + "_Validate: ",
        )

        # switch to evaluation mode
        model.eval()

        #
        n = 0
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)
            if n + images.size(0) > max_num:
                images = images[: max_num - n]
                target = target[: max_num - n]

            with autocast():
                with torch.no_grad():
                    output = forward_classification(images, model, zeroshot_weights)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output, target, topk=(1,))
                    losses.update(loss.item(), images.size(0))
                    top1_org.update(acc1[0].item(), images.size(0))
                    top1_prompt.update(acc1[0].item(), images.size(0))

                torch.cuda.empty_cache()

                # generate adv example
                if args.CW:
                    delta_prompt = attack_CW(
                        model,
                        criterion,
                        images,
                        target,
                        zeroshot_weights,
                        test_stepsize,
                        args.test_numsteps,
                        "l_inf",
                        epsilon=args.test_eps,
                    )
                    attacked_images = images + delta_prompt
                elif args.autoattack:
                    attacked_images = attack_auto(
                        model,
                        images,
                        target,
                        zeroshot_weights,
                        epsilon=args.test_eps,
                        attacks_to_run=attacks_to_run,
                    )
                else:
                    delta_prompt = attack_pgd(
                        model,
                        criterion,
                        images,
                        target,
                        zeroshot_weights,
                        test_stepsize,
                        args.test_numsteps,
                        "l_inf",
                        epsilon=args.test_eps,
                    )
                    attacked_images = images + delta_prompt

                # compute output
                torch.cuda.empty_cache()
                with torch.no_grad():
                    output_adv = forward_classification(attacked_images, model, zeroshot_weights)
                    loss = criterion(output_adv, target)

                # bl attack
                torch.cuda.empty_cache()

                # measure accuracy and record loss
                acc1 = accuracy(output_adv, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_adv_org.update(acc1[0].item(), images.size(0))
                top1_adv_prompt.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            n += images.size(0)
            if n >= max_num:
                break

            if args.debug:
                print("Debug mode. Breaking...")
                break

        torch.cuda.empty_cache()

        print(
            dataset_name
            + " * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} "
            "*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}".format(
                top1_adv_prompt=top1_adv_prompt,
                top1_adv_org=top1_adv_org,
                top1_prompt=top1_prompt,
                top1_org=top1_org,
            )
        )
        acc_all.append(top1_adv_prompt.avg)

        results_dict[dataset_name] = {
            "top1_adv_prompt": top1_adv_prompt.avg,
            "top1_adv_org": top1_adv_org.avg,
            "top1_prompt": top1_prompt.avg,
            "top1_org": top1_org.avg,
        }

        if dataset_name == "cifar10" and top1_org.avg < 0.2:
            print("cifar10 acc is too low. Something is wrong. Exiting...")
            exit()

        # save
        if out_dir is not None and save_name is not None:
            with open(res_path, "w") as f:
                json.dump(results_dict, f, indent=4)

    return np.mean(acc_all), results_dict