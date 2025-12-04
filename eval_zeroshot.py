# eval_zeroshot.py
from argparse import ArgumentParser
import torch
import torch.nn as nn
from utils.model import convert_models_to_fp32
from dataloader import build_val_datasets, get_val_loader_list
from evaluation import build_zeroshot_weights_dict, validate_zeroshot
import numpy as np
import clip


def s2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")

    # model
    parser.add_argument("--model", type=str, default="", choices=["clip", "longclip"])
    parser.add_argument("--imagenet_root", type=str, default="../ILSVRC2012")
    parser.add_argument("--arch", type=str, default="", choices=["vit_b32", "vit_b16", "vit_l14"])

    parser.add_argument("--load_path", type=str, default="", help="path to load model")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")

    # dataset
    parser.add_argument("--root", type=str, default="../data", help="dataset")
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument("--image_size", type=int, default=224, help="image size")

    # test
    parser.add_argument("--test_eps", type=float, default=2, help="momentum")
    parser.add_argument("--test_numsteps", type=int, default=5)
    parser.add_argument("--test_stepsize", type=int, default=1)
    parser.add_argument("--test_n_samples", type=int, default=np.inf)

    # eval
    parser.add_argument("--out_dir_name", type=str, default="eval", help="output directory for evaluation")
    parser.add_argument("--CW", action="store_true")
    parser.add_argument("--autoattack", action="store_true")
    parser.add_argument("--reeval_dataset", type=str, nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attack_norm", type=str, default="l_inf", choices=["l_inf", "l_2"])

    # verbose
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--debug", action="store_true")

    # templates:
    # basic: "a photo of a {c}"
    # default: "a photo of a {c}", "a drawing of a {c}", "a painting of a {c}", ...
    parser.add_argument("--template", type=str, default="basic", choices=["basic", "default"])
    parser.add_argument("--zeroshot_weights_dir", type=str, default="templates/zeroshot-weights")
    parser.add_argument("--overwrite_zeroshot_weights", type=s2bool, default=False)

    return parser.parse_args()


def get_res_name(args, TEST_EPS_INT, TEST_STEPSIZE_INT, TEMPLATE):
    """
    fixed format for result file name
    """
    res_name = "results"
    if args.template == "basic":
        res_name += "_eps{}_numsteps{}_stepsize{}".format(
            TEST_EPS_INT, args.test_numsteps, TEST_STEPSIZE_INT
        )
    else:
        res_name += "_template={}_eps{}_numsteps{}_stepsize{}".format(
            TEMPLATE, TEST_EPS_INT, args.test_numsteps, TEST_STEPSIZE_INT
        )

    if not args.autoattack and not args.CW:
        res_name += "_PGD"
    elif args.CW:
        res_name += "_CW"
    elif args.autoattack:
        res_name += "_autoattack"

    if args.test_n_samples < np.inf:
        res_name += f"_n_samples{args.test_n_samples}"

    return res_name


if __name__ == "__main__":
    args = parse_option()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    if args.arch == "vit_b32":
        model, preprocess = clip.load("ViT-B/32", device, jit=False)
    elif args.arch == "vit_b16":
        model, preprocess = clip.load("ViT-B/16", device, jit=False)
    elif args.arch == "vit_l14":
        model, preprocess = clip.load("ViT-L/14", device, jit=False)
    else:
        raise NotImplementedError

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    # vision encoder
    checkpoint = torch.load(args.load_path, map_location=device)
    try:
        model.module.visual.load_state_dict(checkpoint["vision_encoder_state_dict"])
    except:
        print("The path is vision encoder path")
        model.module.visual.load_state_dict(checkpoint)
    model = model.to(device)

    # val data
    val_dataset_name, val_dataset_list, classnames_dict = build_val_datasets(args)
    val_loader_list = get_val_loader_list(args, val_dataset_list)

    # zeroshot_weights
    zeroshot_weights_dict = build_zeroshot_weights_dict(
        model, device, val_dataset_name, classnames_dict, args
    )

    # evaluation
    criterion = nn.CrossEntropyLoss().to(device)
    TEST_EPS_INT = args.test_eps
    TEST_STEPSIZE_INT = args.test_stepsize
    args.test_eps = args.test_eps / 255.0
    args.test_stepsize = args.test_stepsize / 255.0
    res_name = get_res_name(args, TEST_EPS_INT, TEST_STEPSIZE_INT, TEMPLATE=args.template) 

    acc, results = validate_zeroshot(
        val_loader_list,
        val_dataset_name,
        zeroshot_weights_dict,
        model,
        criterion,
        args,
        max_num=args.test_n_samples,
        out_dir=args.out_dir_name,
        save_name=res_name,
    )