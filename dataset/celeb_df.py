from glob import glob
from os import listdir
from os.path import join
from dataset import AbstractDataset

SPLITS = ["train", "test"]
METHOD = ["YouTube-real", "Celeb-real", "Celeb-synthesis"]


class CelebDF(AbstractDataset):
    """
    Celeb-DF v2 Dataset proposed in "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics"
    """

    def __init__(self, cfg, split, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if split not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {split}.")
        for _ in cfg['method']:
            if _ not in METHOD:
                raise ValueError(f"method should be one of {METHOD}, "
                                f"but found {cfg['method']}.")
        super(CelebDF, self).__init__(cfg, split, seed, transforms, transform, target_transform)
        print(f"Loading data from 'Celeb-DF' of split '{split}'"
              f"\nPlease wait patiently...")
        self.categories = ['original', 'fake']
        self.fpv = cfg.get(f"{split}_fpv", None)
        self.root = cfg['root']
        images_ids = self.__get_images_ids()
        test_ids = self.__get_test_ids()
        train_ids = [images_ids[0] - test_ids[0],
                     images_ids[1] - test_ids[1],
                     images_ids[2] - test_ids[2]]
        train_ids = {
            "YouTube-real": train_ids[0],
            "Celeb-real": train_ids[1],
            "Celeb-synthesis": train_ids[2]
        }
        test_ids = {
            "YouTube-real": test_ids[0],
            "Celeb-real": test_ids[1],
            "Celeb-synthesis": test_ids[2]
        }
        self.images = self.__get_images(test_ids if split == "test" else train_ids, cfg["method"])
        self.targets = list(map(lambda x: 0 if 'real' in x else 1, self.images))
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print(f"\t'{split}' split of CelebDF-v2 {cfg['method']} loaded."
              f"\n\tNum of items: {len(self.images)}. FPV: {self.fpv}.")

    def __get_images_ids(self):
        youtube_real = listdir(join(self.root, 'YouTube-real', 'images'))
        celeb_real = listdir(join(self.root, 'Celeb-real', 'images'))
        celeb_fake = listdir(join(self.root, 'Celeb-synthesis', 'images'))
        return set(youtube_real), set(celeb_real), set(celeb_fake)

    def __get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(join(self.root, "List_of_testing_videos.txt"), "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                number = name.split("/")[-1].split(".")[0]
                if "YouTube-real" in name:
                    youtube_real.add(number)
                elif "Celeb-real" in name:
                    celeb_real.add(number)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(number)
                else:
                    raise ValueError("'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake

    def __get_images(self, ids, methods):
        images = list()
        for i in methods:
            for _ in ids[i]:
                images.extend(glob(join(self.root, i, 'images', _, '*.png')))
        if self.fpv is not None:
            images = self._resample(images, self.fpv)
        return images
