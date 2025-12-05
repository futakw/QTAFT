# utils/io.py
import sys
import os
import shutil
import torch
from collections import OrderedDict

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        if "tqdm" not in message:
            self.stdout.write(message)
            self.file.write(message)
            self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def save_checkpoint(state, args, filename="checkpoint.pth.tar"):
    savefile = os.path.join(args.model_folder, filename)
    torch.save(state, savefile)


def fix_model_state_dict(state_dict):
    # DataParallel("module.")とwrapper("net.")を剥がす
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    state_dict = new_state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k.replace("net.", "")] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
