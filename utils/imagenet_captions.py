# https://github.com/williamFalcon/pytorch-imagenet-dataset

import os
import json
import time

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch.utils.data as data
from torchvision.datasets.folder import make_dataset, find_classes, default_loader, IMG_EXTENSIONS
from torchvision import datasets

from utils.text_prompts import load_imagenet_folder2name, refine_classname

IMAGENET_CLASSES_NAMES_PATH = "data/imagenet_classes_names.txt"

# https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
class ImageNetCaptions(datasets.VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        image_caption_root: str = None,
        # annot_type: str = None,
    ) -> None:
        self.root = root

        st = time.time()
        print("========== Finding classes... ==========")
        classes, class_to_idx = self.find_classes(self.root)
        print("Classes found: ", len(classes))
        print("Time taken: ", time.time() - st)

        st = time.time()
        print("========== Making dataset... ==========")
        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        print("Dataset made: ", len(samples))
        print("Time taken: ", time.time() - st)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        self.image_captions = {}
        self.file2caption = {}
        for _class in classes:
            image_cap_path = os.path.join(image_caption_root, _class + ".json")
            with open(image_cap_path, "r") as f:
                image_captions = json.load(f)
            self.image_captions[_class] = image_captions
            for k, v in image_captions.items():
                k = k.split("/")[-1]
                self.file2caption[k] = v
        # self.file2caption = {d["filename"]: t for t in self.image_captions.values() for d in t}

        n_found = 0
        n_not_found = 0
        for i, (path, target) in enumerate(self.samples):
            file_name = os.path.basename(path)
            if file_name not in self.file2caption:
                print("Warning: No caption found for image: ", path)
                self.file2caption[file_name] = ""
                n_not_found += 1
            else:
                n_found += 1
        print("Captions found: ", n_found)
        print("Captions not found: ", n_not_found)
        assert n_not_found < 500 # sanity check
        if n_not_found > 0:
            print("########## WARNING: Some captions not found ##########")

        folder2name = load_imagenet_folder2name(IMAGENET_CLASSES_NAMES_PATH)
        new_class_names = []
        for each in classes:
            new_class_names.append(folder2name[each])

        self.class_names = refine_classname(new_class_names)

    @staticmethod
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
                An error is raised on empty folders if False (default).

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory,
            class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        caption = self.file2caption[os.path.basename(path)]

        return sample, target, caption, index

    def __len__(self) -> int:
        return len(self.samples)
