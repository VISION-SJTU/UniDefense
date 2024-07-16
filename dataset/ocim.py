import copy
import torch
from os.path import join
from dataset import AbstractDataset

DATASET = ["O", "C", "I", "M"]
SPLIT = ["train", "dev", "test"]
LABEL = ["real", "fake", "both"]


class OCIMSubDataset(AbstractDataset):
    """
    OCIM Protocol for Domain Generalization for Face Anti-Spoofing
    """

    def __init__(self, cfg, split, label, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if split not in SPLIT:
            raise ValueError(f"split should be one of '{SPLIT}', but found '{split}'.")
        if label not in LABEL:
            raise ValueError(f"label should be one of '{LABEL}', but found '{label}'.")
        dataset = cfg[split + "_dataset"]
        if dataset not in DATASET:
            raise ValueError(f"available dataset should be one of '{DATASET}', "
                             f"but '{dataset}' is not supported.")
        super(OCIMSubDataset, self).__init__(
            cfg, split, seed, transforms, transform, target_transform)
        self.categories = ["real", "attack"]
        d_list_loc = join(self.root, cfg[dataset + "_root"], "lists")
        self.fpv = cfg.get(f"{split}_fpv", None)
        if label == "both":
            real_list = torch.load(join(d_list_loc, f"real_5points.pickle"))
            if self.fpv is not None:
                real_list = self._resample(real_list, self.fpv)
            self.targets.extend([0] * len(real_list))
            fake_list = torch.load(join(d_list_loc, f"fake_5points.pickle"))
            if self.fpv is not None:
                fake_list = self._resample(fake_list, self.fpv)
            self.targets.extend([1] * len(fake_list))
            d_list = real_list + fake_list
        else:
            d_list = torch.load(join(d_list_loc, f"{label}_5points.pickle"))
            if self.fpv is not None:
                d_list = self._resample(d_list, self.fpv)
            self.targets.extend([0 if label == "real" else 1] * len(d_list))
        self.images.extend(d_list)
        assert len(self.images) == len(self.targets), "dataset corrupts."
        print(f"==> '{split}' split with label '{label}' of dataset '{dataset}' loaded. "
              f"Num of items: {len(self.images)}. FPV: {self.fpv}.")

class OCIMDataset(object):
    def __init__(self, cfg, split, seed=2022):
        self.datasets = list()
        datasets = cfg[split + "_dataset"]
        self.num_domains = len(datasets)
        for ds in datasets:
            ds_cfg = copy.deepcopy(cfg)
            ds_cfg[split + "_dataset"] = ds
            self.datasets.append(OCIMSubDataset(ds_cfg, split, "real", seed))
            self.datasets.append(OCIMSubDataset(ds_cfg, split, "fake", seed))
