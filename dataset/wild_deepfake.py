import torch
import numpy as np
from os.path import join
from dataset import AbstractDataset

SPLITS = ["train", "test"]
METHOD = ["real", "fake"]


class WildDeepfake(AbstractDataset):
    """
    Wild Deepfake Dataset proposed in "WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection"
    """

    def __init__(self, cfg, split, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if split not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {split}.")
        for _ in cfg['method']:
            if _ not in METHOD:
                raise ValueError(f"method should be one of {METHOD}, "
                                f"but found {cfg['method']}.")
        super(WildDeepfake, self).__init__(cfg, split, seed, transforms, transform, target_transform)
        print(f"Loading data from 'WildDeepfake' of split '{split}'"
              f"\nPlease wait patiently...")
        self.categories = ['original', 'fake']
        self.fpv = cfg.get(f"{split}_fpv", None)
        self.root = cfg['root']
        self.images, self.targets = self.__get_images(cfg["method"])
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print(f"\t'{split}' split of WildDeepfake {cfg['method']} loaded."
              f"\n\tNum of items: {len(self.images)}. FPV: {self.fpv}.")

    def __get_images(self, methods):
        images = list()
        targets = list()
        for _ in methods:
            imgs = torch.load(join(self.root, self.split, f"{_}.pickle"))
            if self.fpv is not None:
                imgs = self._resample(imgs, self.fpv)
            images.extend(imgs)
            tgts = [torch.tensor(0 if _ == "real" else 1)] * len(imgs)
            targets.extend(tgts)
        return images, targets

    def __getitem__(self, index):
        path = join(self.root, self.split, self.images[index])
        tgt = self.targets[index]
        return path, tgt
