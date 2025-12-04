# utils/text_prompts.py
import torch

def refine_classname(class_names):
    return [
        c.lower().replace("_", " ").replace("-", " ").replace("/", " ")
        for c in class_names
    ]


def load_imagenet_folder2name(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            split_name = line.strip().split()
            if len(split_name) < 3:
                continue
            wnid = split_name[0]
            cat_name = split_name[2]
            mapping[wnid] = cat_name
    return mapping


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]


def get_text_prompts_train(args, train_dataset, template="This is a photo of a {}"):
    class_names = train_dataset.classes
    if args.dataset == "ImageNet":
        folder2name = load_imagenet_folder2name("imagenet_classes_names.txt")
        class_names = [folder2name[c] for c in class_names]
    class_names = refine_classname(class_names)
    return [template.format(label) for label in class_names]


def get_text_prompts_val(val_dataset_list, val_dataset_name, template="This is a photo of a {}"):
    texts_list = []
    for cnt, ds in enumerate(val_dataset_list):
        if hasattr(ds, "clip_prompts"):
            texts_tmp = ds.clip_prompts
        else:
            class_names = ds.classes
            if val_dataset_name[cnt] == "ImageNet":
                from .text_prompts import load_imagenet_folder2name
                folder2name = load_imagenet_folder2name("imagenet_classes_names.txt")
                class_names = [folder2name[c] for c in class_names]
            class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list
