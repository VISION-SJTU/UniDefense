import cv2
import lmdb
import torch
import numpy as np
from os.path import join
from typing import List
from typing_extensions import Literal
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import (
    OneOf,
    Resize,
    Compose,
    Normalize,
    GaussNoise,
    ColorJitter,
    GaussianBlur,
    ImageCompression,
    RandomBrightnessContrast
)
from albumentations.pytorch.transforms import ToTensorV2


METHOD =[
    'FFpp-DF', 'FFpp-F2F', 'FFpp-FS', 'FFpp-NT', 'FFpp-Real',
    'CDF-Fake', 'CDF-Real',
    'SeqDF-Fake', 'SeqDF-Real',
    'HQ-Flexiblemask', 'HQ-Glasses', 'HQ-Makeup', 'HQ-Mannequin',
    'HQ-Papermask', 'HQ-Replay', 'HQ-Rigidmask', 'HQ-Tattoo', 'HQ-Real',
    'OULU-Fake', 'OULU-Real',
    'SiWMv2-Fake', 'SiWMv2-Real'
]
SPLIT = ['train', 'val', 'test']


class AbstractUniAttack(VisionDataset):
    """All the considered datasets are in LMDB format."""
    def __init__(self, cfg, split, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractUniAttack, self).__init__(
            cfg['root'], transforms=transforms, transform=transform, target_transform=target_transform)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        print("Using dataset: 'UniAttack'")

        self.images = list()
        self.targets = list()
        self.split = split

        # Dummy path
        self.root = cfg["root"]

        # Path to four considered datasets
        self.FFpp_root = cfg["FFpp_root"]
        self.CDF_root = cfg["CDF_root"]
        self.SeqDF_root = cfg["SeqDF_root"]
        self.HQ_root = cfg["HQ_root"]
        self.OULU_root = cfg["OULU_root"]
        self.SiWMv2_root = cfg["SiWMv2_root"]

        if self.FFpp_root is not None:
            self.FFpp_lmdb = lmdb.open(join(self.FFpp_root, "lmdb", 'FaceForensics++'), readonly=True, max_readers=512)
            self.FFpp_lmdb = self.FFpp_lmdb.begin(write=False)

        if self.CDF_root is not None:
            self.CDF_lmdb = lmdb.open(join(self.CDF_root, "lmdb", 'Celeb-DF'), readonly=True, max_readers=512)
            self.CDF_lmdb = self.CDF_lmdb.begin(write=False)

        if self.SeqDF_root is not None:
            self.SeqDF_lmdb = lmdb.open(join(self.SeqDF_root, "lmdb", 'Seq-DeepFake'), readonly=True, max_readers=512)
            self.SeqDF_lmdb = self.SeqDF_lmdb.begin(write=False)

        if self.HQ_root is not None:
            self.HQ_lmdb = lmdb.open(join(self.HQ_root, "lmdb", 'HQ_WMCA'), readonly=True, max_readers=512)
            self.HQ_lmdb = self.HQ_lmdb.begin(write=False)

        if self.OULU_root is not None:
            self.OULU_lmdb = lmdb.open(join(self.OULU_root, "lmdb", 'Oulu_NPU'), readonly=True, max_readers=512)
            self.OULU_lmdb = self.OULU_lmdb.begin(write=False)

        if self.SiWMv2_root is not None:
            self.SiWMv2_lmdb = lmdb.open(join(self.SiWMv2_root, "lmdb", 'SiW-Mv2'), readonly=True, max_readers=512)
            self.SiWMv2_lmdb = self.SiWMv2_lmdb.begin(write=False)

        if self.transforms is None:
            self.transforms = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg[self.split + '_transforms']] +
                [ToTensorV2()]
            )
        # Rewrite for Protocol I (Distorted)
        if self.split == "test" and cfg.get("distorted", False):
            compose = Compose([
                Resize(
                    height=cfg['train_transforms'][0]['params']['height'],
                    width=cfg['train_transforms'][0]['params']['width'],
                ),
                OneOf([
                    ImageCompression(quality_lower=50, quality_upper=60, p=0.2),
                    GaussianBlur(blur_limit=(9, 11), p=0.2),
                    GaussNoise(var_limit=(10, 20), p=0.2),
                    RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.5, p=0.2),
                    ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.0, p=0.2)
                ], p=1),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
            print("==> Using distorted test transforms.")
            self.transforms = compose

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        tgt = self.targets[index]
        return path, tgt

    @staticmethod
    def _resample(list_file, frames_per_video):
        video_dict = dict()
        for i in list_file:
            name = i.split(" ")[0]
            video_path = name.rsplit('/', 1)[0]
            if video_path not in video_dict:
                video_dict.update({video_path: [i]})
            else:
                video_dict[video_path].append(i)
        # print(f'total videos: {len(video_dict)}')
        resample_list = list()
        for i, j in video_dict.items():
            if len(j) <= frames_per_video:  # do not resample in this case
                resample = j
            else:
                resample = np.random.choice(j, frames_per_video, replace=False)
                resample = sorted(resample, key=lambda x: x.split(" ")[0])
            resample_list.extend(resample)
        return resample_list

    @staticmethod
    def _add_face_margin(x, y, w, h, margin=0.5):
        x_marign = int(w * margin / 2)
        y_marign = int(h * margin / 2)

        x1 = x - x_marign
        x2 = x + w + x_marign
        y1 = y - y_marign
        y2 = y + h + y_marign

        return x1, x2, y1, y2

    def _convert_to_str(self, img_path, feature, postfix="jpg"):
        if 'manipulated_sequences' in img_path or 'original_sequences' in img_path:
            out_path = img_path
        elif 'Celeb-real' in img_path or 'Celeb-synthesis' in img_path or 'YouTube-real' in img_path:
            out_path = img_path
        elif 'Seq-DeepFake' in img_path:
            out_path = img_path[:-4] + f"_{feature}.jpg"
        elif 'Oulu_NPU' in img_path:
            out_path = img_path.replace('Oulu_NPU', f'Oulu_NPU_{feature}')
        elif 'HQ_WMCA' in img_path:
            out_path = img_path.replace('.jpg', f'_{feature}.jpg')
        elif 'SiW-Mv2' in img_path:
            out_path = img_path[:-4] + f"_{feature}.jpg"
        else:
            raise ValueError("Image path corrupted.")
        out_path = out_path.replace(".jpg", f".{postfix}")
        return out_path

    def _get_actual_root(self, img_path):
        if 'manipulated_sequences' in img_path or 'original_sequences' in img_path:
            return self.FFpp_root
        elif 'Celeb-real' in img_path or 'Celeb-synthesis' in img_path or 'YouTube-real' in img_path:
            return self.CDF_root
        elif 'Seq-DeepFake' in img_path:
            return self.SeqDF_root
        elif 'Oulu_NPU' in img_path:
            return self.OULU_root
        elif 'HQ_WMCA' in img_path:
            return self.HQ_root
        elif 'SiW-Mv2' in img_path:
            return self.SiWMv2_root
        else:
            raise ValueError(f"Image path: '{img_path}' corrupted.")

    def _get_actual_lmdb(self, img_path):
        if 'manipulated_sequences' in img_path or 'original_sequences' in img_path:
            return self.FFpp_lmdb
        elif 'Celeb-real' in img_path or 'Celeb-synthesis' in img_path or 'YouTube-real' in img_path:
            return self.CDF_lmdb
        elif 'Seq-DeepFake' in img_path:
            return self.SeqDF_lmdb
        elif 'Oulu_NPU' in img_path:
            return self.OULU_lmdb
        elif 'HQ_WMCA' in img_path:
            return self.HQ_lmdb
        elif 'SiW-Mv2' in img_path:
            return self.SiWMv2_lmdb
        else:
            raise ValueError(f"Image path: '{img_path}' not supported for lmdb.")

    def load_item(self, items, labels,
                  margin=None,
                  crop="nocrop", dataset_label_map=None):
        images = list()
        path = list()
        dataset_labels = list()
        for item, label in zip(items, labels):
            contents = item.split(" ")
            img_path = contents[0]
            path.append(img_path)
            # process domain label for some methods
            dataset_label = self._get_actual_root(img_path)
            if dataset_label_map is not None:
                dataset_labels.append(dataset_label_map[dataset_label])
            
            if crop == "nocrop":
                # We have cropped the image
                crop_path = self._convert_to_str(img_path, "crop")
            else:
                # Crop the image on-the-fly
                crop_path = img_path
            img_bin = self._get_actual_lmdb(img_path).get(crop_path.encode())
            img_buf = np.frombuffer(img_bin, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if crop == "nocrop" or dataset_label in [self.FFpp_root, self.CDF_root]:
                max_h, max_w = img.shape[:2]
                x1, x2, y1, y2 = 0, max_w, 0, max_h
            elif crop == "4p":
                # crop face from bbox using 4-point coordinates
                x, y, w, h = [int(_) for _ in contents[2:6]]
                if isinstance(margin, float):
                    # A fixed margin
                    pass
                else:
                    # Random initialized a margin
                    random_mgn = np.random.randint(int(margin[0] * 10),
                                                   int(margin[1] * 10))
                    margin = random_mgn / 10.0
                x1, x2, y1, y2 = self._add_face_margin(x, y, w, h, margin)
            else:
                raise ValueError(f"not supported crop version '{crop}'.")

            max_h, max_w = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(max_w, x2)
            y2 = min(max_h, y2)
            img = img[y1:y2, x1:x2]

            results = self.transforms(image=img)
            image = results['image']

            images.append(image)

        out = {
            'images': torch.stack(images, dim=0),
            'path': path,
            'dataset_labels': torch.tensor(dataset_labels, dtype=torch.long) \
                if len(dataset_labels) > 0 else None
        }

        return out


class UniAttack(AbstractUniAttack):
    """
    UniAttack benchmark dataset for joint detection of face forgery attacks and face spoofing attacks.
    """

    def __init__(self, cfg, split, methods, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if split not in SPLIT:
            raise ValueError(f"split should be one of '{SPLIT}', but found '{split}'.")
        for _ in methods:
            if _ not in METHOD:
                raise ValueError(f"method should be one of {METHOD}, but found {methods}.")
        super(UniAttack, self).__init__(
            cfg, split, seed, transforms, transform, target_transform)
        print(f"\tLoading data from 'UniAttack {methods}' of split '{split}' \n"
              f"\tPlease wait patiently...")

        self.categories = ['original', 'fake']
        self.real_fpv = cfg.get(f"{split}_real_fpv", None)
        self.fake_fpv = cfg.get(f"{split}_fake_fpv", None)
        for method in methods:
            ds, me = method.split("-")
            img, tgt = getattr(self, f"_load_{ds.lower()}")(me)
            self.images.extend(img)
            self.targets.extend(tgt)
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print(f"\t'{split}' split of UniAttack loaded, with #SubMethod = {len(methods)}.\n"
              f"\tNum of items: {len(self.images)}. Real FPV: {self.real_fpv}, Fake FPV: {self.fake_fpv}.")
        
    
    def _load_ffpp(self, method: Literal["DF", "F2F", "FS", "NT", "Real"]):
        ffpp_dict = {
            'DF': 'Deepfakes',
            'F2F': 'Face2Face',
            'FS': 'FaceSwap',
            'NT': 'NeuralTextures',
            'Real': 'original_sequences'
        }
        indices = list()
        pre_indices = join(self.FFpp_root, 'pickle_files', self.split + "_c23.pickle")
        pre_indices = torch.load(pre_indices)
        for (path, _) in pre_indices:
            if ffpp_dict[method] in path:
                indices.append(path)
        
        fpv = self.real_fpv if method == "Real" else self.fake_fpv
        if fpv is not None:
            indices = self._resample(indices, fpv)
        targets = [0 if method == "Real" else 1] * len(indices)
        return indices, targets
    
    def _load_cdf(self, method: Literal["Real", "Fake"]):
        indices = list()
        candidate_paths = torch.load(join(self.CDF_root, "pickle_files", f"{self.split}.pickle"))

        if method == "Real":
            for _ in candidate_paths:
                if "YouTube-real" in _ or "Celeb-real" in _:
                    indices.append(_)
        else:
            for _ in candidate_paths:
                if "Celeb-synthesis" in _:
                    indices.append(_)
        
        fpv = self.real_fpv if method == "Real" else self.fake_fpv
        if fpv is not None:
            indices = self._resample(indices, fpv)
        targets = [0 if method == "Real" else 1] * len(indices)
        return indices, targets

    def _load_seqdf(self, method: Literal["Real", "Fake"]):
        indices = torch.load(join(
            self.SeqDF_root, "pickle_files", f"{self.split}_{method.lower()}.pickle"))
        
        # Seq-DeepFake essentially a frame-level dataset
        # So do not use frame per video to resample
        targets = [0 if method == "Real" else 1] * len(indices)
        return indices, targets
    
    def _load_hq(self, method: Literal[
        "Real",
        "Flexiblemask",
        "Glasses",
        "Makeup",
        "Mannequin",
        "Papermask",
        "Replay",
        "Rigidmask",
        "Tattoo"
    ]):
        hq_split_dict = {
            "train": "train",
            "val": "dev",
            "test": "eval"
        }
        indices = list()
        protocol = join(self.HQ_root, "PROTOCOL-grand_test-curated.csv")
        record = torch.load(join(self.HQ_root, "record.pickle"))
        
        with open(protocol, "r", encoding="utf-8") as file:
            contents = file.readlines()
        collected = list()
        if method == "Real":
            for line in contents:
                items = line.strip().split(",")
                if items[1] == "0" and items[-1] == hq_split_dict[self.split]:
                    collected.append(items)
        else:
            for line in contents:
                items = line.strip().split(",")
                if items[2] == f"attack/{method}" and items[-1] == hq_split_dict[self.split]:
                    collected.append(items)
        
        # load images in essence
        for items in collected:
            directory = items[0].split("/")[-1]
            img_list = record[directory]
            indices.extend(img_list)
        
        fpv = self.real_fpv if method == "Real" else self.fake_fpv
        if fpv is not None:
            indices = self._resample(indices, fpv)
        targets = [0 if method == "Real" else 1] * len(indices)
        return indices, targets
    
    def _load_oulu(self, method: Literal["Real", "Fake"]):
        oulu_split_dict = {
            "train": "Train_files",
            "val": "Dev_files",
            "test": "Test_files"
        }
        data_list_loc = join(self.OULU_root, "lists",
                             f"{method.lower()}_5points.pickle")
        data_list = torch.load(data_list_loc)
        indices = list(filter(lambda x: oulu_split_dict[self.split] in x, data_list))

        fpv = self.real_fpv if method == "Real" else self.fake_fpv
        if fpv is not None:
            indices = self._resample(indices, fpv)
        targets = [0 if method == "Real" else 1] * len(indices)
        return indices, targets

    def _load_siwmv2(self, method: Literal["Real", "Fake"]):
        indices = list()
        label = "live" if method == "Real" else "all"
        data_list_loc = join(self.SiWMv2_root, "lists",
                             f"{self.split.lower()}list_{label}.pickle")
        
        indices = torch.load(data_list_loc)

        fpv = self.real_fpv if method == "Real" else self.fake_fpv
        if fpv is not None:
            indices = self._resample(indices, fpv)
        targets = [0 if method == "Real" else 1] * len(indices)
        return indices, targets
