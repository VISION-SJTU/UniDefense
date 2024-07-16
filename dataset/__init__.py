from .abstract_dataset import AbstractDataset
from .faceforensics import FaceForensics
from .celeb_df import CelebDF
from .wild_deepfake import WildDeepfake
from .ocim import OCIMDataset
from .uniattack import UniAttack


LOADERS = {
    "FFpp": FaceForensics,
    "CDF": CelebDF,
    "WDF": WildDeepfake,
    "OCIM": OCIMDataset,
    "UniAttack": UniAttack,
}


def get_dataset(name="FFpp"):
    assert name in LOADERS, f"Dataset '{name}' not found."
    print(f"Using dataset: '{name}'")
    return LOADERS[name]
