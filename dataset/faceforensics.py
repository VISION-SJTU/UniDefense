import torch
from os.path import join
from dataset import AbstractDataset

METHOD = ['Origin', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter', 'DeeperForensics']
SPLIT = ['train', 'val', 'test']
COMP2NAME = {'c0': 'raw', 'c23': 'c23', 'c40': 'c40'}
SOURCE_MAP = {
    'youtube': 2, 'Deepfakes': 3, 'Face2Face': 4, 'FaceSwap': 5, 'NeuralTextures': 6,
    'FaceShifter': 7, 'DeeperForensics': 8
}


class FaceForensics(AbstractDataset):
    """
    FaceForensics++ Dataset proposed in "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(self, cfg, split, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if split not in SPLIT:
            raise ValueError(f"split should be one of '{SPLIT}', but found '{split}'.")
        for _ in cfg['method']:
            if _ not in METHOD:
                raise ValueError(f"method should be one of {METHOD}, "
                                f"but found {cfg['method']}.")
        if cfg['compression'] not in COMP2NAME.keys():
            raise ValueError(f"compression should be one of {COMP2NAME.keys()}, "
                             f"but found {cfg['compression']}.")
        super(FaceForensics, self).__init__(
            cfg, split, seed, transforms, transform, target_transform)
        print(f"\tLoading data from 'FF++ {cfg['method']}' of split '{split}' "
              f"and compression '{cfg['compression']}'\n\tPlease wait patiently...")

        self.categories = ['original', 'fake']
        self.fpv = cfg.get(f"{split}_fpv", None)

        indices = list()
        # load the path of dataset images from pre-stored pickle
        pre_indices = join(self.root, 'pickle_files', split + "_" + cfg['compression'] + ".pickle")
        pre_indices = torch.load(pre_indices)
        for (path, _) in pre_indices:
            if METHOD[0] in cfg["method"] and 'original' in path:
                indices.append(path)
            for i in METHOD[1:]:
                if i in cfg['method'] and i in path:
                    indices.append(path)
        if self.fpv is not None:
            indices = self._resample(indices, self.fpv)

        self.images = indices
        self.targets = list(map(lambda x: 0 if 'original_sequences' in x else 1, self.images))
        assert len(self.images) == len(self.targets), "dataset corrupts."
        print(f"\t'{split}' split of FaceForensics {cfg['compression']}-{cfg['method']} loaded."
              f"\n\tNum of items: {len(self.images)}. FPV: {self.fpv}.")
