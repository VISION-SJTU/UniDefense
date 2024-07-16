import cv2
import lmdb
import torch
import numpy as np
from os.path import join
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2


class AbstractDataset(VisionDataset):
    def __init__(self, cfg, split, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractDataset, self).__init__(cfg['root'], transforms=transforms,
                                              transform=transform, target_transform=target_transform)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        self.images = list()
        self.targets = list()
        self.split = split
        self.root = cfg["root"]
        self.use_lmdb = cfg.get("use_lmdb", True)
        
        ds = cfg.get(f"{split}_dataset")
        if self.use_lmdb:
            if ds is not None:
                self.env = lmdb.open(join(self.root, "lmdb", cfg[ds + '_root']), readonly=True, max_readers=512)
            else:
                self.env = lmdb.open(join(self.root, "lmdb", cfg["lmdb"]), readonly=True, max_readers=512)
            self.lmdb = self.env.begin(write=False)
        else:
            self.lmdb = None

        print(f"==> Use lmdb {self.use_lmdb}.")
        if self.transforms is None:
            self.transforms = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg[self.split + '_transforms']] +
                [ToTensorV2()]
            )

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
        if 'replayattack' in img_path:
            out_path = img_path.replace('replayattack', f'replayattack_{feature}')
        elif 'CASIA_database' in img_path:
            out_path = img_path.replace('CASIA_database', f'CASIA_database_{feature}')
        elif 'MSU-MFSD' in img_path:
            out_path = img_path.replace('MSU-MFSD', f'MSU-MFSD_{feature}')
        elif 'Oulu_NPU' in img_path:
            out_path = img_path.replace('Oulu_NPU', f'Oulu_NPU_{feature}')
        elif 'HQ_WMCA' in img_path:
            out_path = img_path.replace('.jpg', f'_{feature}.jpg')
        elif "Siw-MV2" in self.root:
            out_path = img_path.replace('.jpg', f'_{feature}.jpg')
        else:
            raise ValueError("Image path corrupted.")
        out_path = out_path.replace(".jpg", f".{postfix}")
        return out_path

    def load_item(
        self,
        items,
        labels,
        margin=None,
        crop="4p"
    ):
        images = list()
        path = list()
        for item, label in zip(items, labels):
            contents = item.split(" ")
            img_path = contents[0]
            path.append(img_path)
            
            if self.use_lmdb:
                # mainly for face anti-spoofing datasets
                crop_path = self._convert_to_str(img_path, "crop")
                img_bin = self.lmdb.get(crop_path.encode())
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            else:
                # mainly for face forgery detection datasets
                img = cv2.imread(join(self.root, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if crop == "4p":
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
            elif crop == "nocrop":
                max_h, max_w = img.shape[:2]
                x1, x2, y1, y2 = 0, max_w, 0, max_h
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
            'path': path
        }

        return out
