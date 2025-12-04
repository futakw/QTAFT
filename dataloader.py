import os
import json
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    STL10,
    Caltech101,
    Caltech256,
    Country211,
    DTD,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageFolder,
    OxfordIIITPet,
    PCAM,
    SUN397,
    StanfordCars,
)
from data import imagenetv2
from data.imagenet_variants_classes import (
    all_imagenet_wordnet_ids,
    imagenet_a_wnids,
    imagenet_o_wnids,
    imagenet_r_wnids,
)
from utils.imagenet_captions import ImageNetCaptions
from utils.text_prompts import refine_classname

DATASET_EN_CLASSNAMES_PATH = "data/en_classnames.json"

def build_train_loader(args, imagenet_root, preprocess224):
    assert args.dataset in ["ImageNet", "ImageNet-100"], "Only ImageNet is supported for training"

    if args.dataset == "ImageNet":
        train_root = os.path.join(imagenet_root, "train")
    else:
        train_root = "../data/ImageNet-100/train"

    train_dataset = ImageNetCaptions(
        train_root,
        transform=preprocess224,
        image_caption_root=args.image_caption_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,  # due to memory issue
        num_workers=args.num_workers,
        shuffle=True,
        sampler=None,
    )
    return train_dataset, train_loader


def build_val_datasets(args):
    # 旧 eval と同じ val_dataset_name
    val_dataset_name = [
        "cifar10", "cifar100", "STL10", "SUN397", "StanfordCars", "Food101",
        "oxfordpet", "flowers102", "dtd", "EuroSAT", "fgvc_aircraft", "PCAM",
        "ImageNet", "Caltech101",
        "ImageNet-S", "ImageNet-R", "ImageNet-v2", "ImageNet-O", "ImageNet-A",
    ]

    # transforms も旧 eval と同じ
    preprocess = transforms.Compose([transforms.ToTensor()])
    preprocess224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with open(DATASET_EN_CLASSNAMES_PATH, "r") as f:
        classnames_json = json.load(f)

    val_dataset_list = []
    classnames_dict = {}

    imgnet_full = args.imagenet_root

    to_dl = False
    for each in val_dataset_name:
        print("Loading Validation Dataset", each)
        if each == "cifar10":
            ds = CIFAR10(args.root, transform=preprocess, download=to_dl, train=False)
        elif each == "cifar100":
            ds = CIFAR100(args.root, transform=preprocess, download=to_dl, train=False)
            ds.classes = [c.replace("_", " ") for c in ds.classes]
        elif each == "Caltech101":
            ds = Caltech101(args.root, target_type="category", transform=preprocess224, download=to_dl)
            ds.classes = classnames_json["caltech101"]
            ds.classes = refine_classname(ds.classes)
            rename_dict = {
                "flamingo head": "head of a flamingo",
                "cougar body": "body of a cougar cat",
                "car side": "side of a car",
                "person": "centered face",
                "cougar face": "face of a cougar cat",
                "crocodile head": "head of a crocodile",
                "motorbikes": "motorbike",
                "airplanes": "airplane",
                "leopards": "leopard",
                "snoopy": "snoopy (cartoon beagle)",
                "yin yang": "yin and yang symbol",
            }
            new_classes = []
            for cl in ds.classes:
                if cl in rename_dict:
                    cl = rename_dict[cl]
                new_classes.append(cl)
            ds.classes = new_classes
        elif each == "Caltech256":
            ds = Caltech256(args.root, transform=preprocess224, download=to_dl)
        elif each == "PCAM":
            ds = PCAM(args.root, split="test", transform=preprocess224, download=to_dl)
            ds.classes = classnames_json["pcam"]
        elif each == "STL10":
            ds = STL10(args.root, split="test", transform=preprocess, download=to_dl)
        elif each == "SUN397":
            ds = SUN397(args.root, transform=preprocess224, download=to_dl)
            # refine classname for CuPL
            new_classnames = []
            for cl in ds.classes:
                if "/" in cl:
                    cl = cl + ")"
                    cl = cl.replace("/", " (")
                cl = cl.replace("_", " ")
                new_classnames.append(cl)
            ds.classes = new_classnames
        elif each == "StanfordCars":
            ds = StanfordCars(args.root, split="test", transform=preprocess224, download=to_dl)
            new_classnames = []
            for cl in ds.classes:
                if cl == "Ram C/V Cargo Van Minivan 2012":
                    cl = "Ram CV Cargo Van Minivan 2012"
                new_classnames.append(cl)
            ds.classes = new_classnames
        elif each == "Food101":
            ds = Food101(args.root, split="test", transform=preprocess224, download=to_dl)
            new_classnames = []
            for cl in ds.classes:
                cl = cl.replace("_", " ")
                new_classnames.append(cl)
            ds.classes = new_classnames
        elif each == "oxfordpet":
            ds = OxfordIIITPet(args.root, split="test", transform=preprocess224, download=to_dl)
            ds.classes = refine_classname(ds.classes)
        elif each == "EuroSAT":
            ds = EuroSAT(args.root, transform=preprocess224, download=to_dl)
            ds.classes = classnames_json["eurosat"]
        elif each == "flowers102":
            ds = Flowers102(args.root, split="test", transform=preprocess224, download=to_dl)
            ds.classes = classnames_json["flowers"]
        elif each == "Country211":
            ds = Country211(args.root, split="test", transform=preprocess224, download=to_dl)
            ds.classes = classnames_json["country211"]
        elif each == "dtd":
            ds = DTD(args.root, split="test", transform=preprocess224, download=to_dl)
        elif each == "fgvc_aircraft":
            ds = FGVCAircraft(args.root, split="test", transform=preprocess224, download=to_dl)
        elif each == "hateful_memes":
            ds = HatefulMemes(
                args.root, splits=["test_seen", "test_unseen"], transform=preprocess224_interpolate
            )
        elif each == "ImageNet":
            ds = torchvision.datasets.ImageFolder(os.path.join(imgnet_full, "val"), transform=preprocess224)
            ds.classes = classnames_json["imagenet1k"]
        elif each == "ImageNet-100":
            ds = torchvision.datasets.ImageFolder("../data/ImageNet-100/val", transform=preprocess224)
        elif each == "ImageNet-S":
            ds = ImageFolder(root="../data/sketch", transform=preprocess224)
            ds.classes = classnames_json["imagenet1k"]
        elif each == "ImageNet-R":
            imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_imagenet_wordnet_ids]
            ds = ImageFolder(root="../data/imagenet-r", transform=preprocess224)
            ds.classes = classnames_json["imagenet1k"]
            ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_r_mask) if mask]
        elif each == "ImageNet-v2":
            ds = imagenetv2.ImageNetV2Dataset(
                variant="matched-frequency", transform=preprocess224, location="../data"
            )
            ds.classes = classnames_json["imagenet1k"]
        elif each == "ImageNet-O":
            ds = ImageFolder(root="../data/imagenet-o", transform=preprocess224)
            ds.classes = classnames_json["imagenet1k"]
            imagenet_o_mask = [wnid in set(imagenet_o_wnids) for wnid in all_imagenet_wordnet_ids]
            ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_o_mask) if mask]
        elif each == "ImageNet-A":
            ds = ImageFolder(root="../data/imagenet-a", transform=preprocess224)
            ds.classes = classnames_json["imagenet1k"]
            imagenet_a_mask = [wnid in set(imagenet_a_wnids) for wnid in all_imagenet_wordnet_ids]
            ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_a_mask) if mask]
        else:
            raise NotImplementedError

        # append
        val_dataset_list.append(ds)
        classnames_dict[each] = ds.classes

    return val_dataset_name, val_dataset_list, classnames_dict


def get_val_loader_list(args, val_dataset_list):
    val_loader_list = []
    for ds in val_dataset_list:
        if args.test_n_samples < len(ds):
            subset_indices = list(range(len(ds)))
            g = torch.Generator()
            g.manual_seed(42)  # for reproducibility
            subset_indices = torch.randperm(len(subset_indices), generator=g).tolist()
            subset_indices = subset_indices[: args.test_n_samples]
            val_sampler = SubsetRandomSampler(subset_indices)
        else:
            val_sampler = None

        val_loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            pin_memory=False,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=val_sampler,
        )
        val_loader_list.append(val_loader)
    return val_loader_list