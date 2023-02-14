import os
import re
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from typing import Tuple
import torch

class CityScapes(Dataset):
    ground_truth = 'gtFine'
    image_left = "leftImg8bit"
    image_right = "rightImg8bit"

    def __init__(
            self,
            root: Path = Path("/data/Datastore/Cityscapes"),
            type: str = "train",
            image_side: str = "left",
            gt_type: str = "labelIds"
    ):
        assert type in ['train', 'test', 'val']
        assert image_side in ['left', 'right']
        assert gt_type in ["labelIds", "polygons", "instanceIds", "color"]
        self.image_side = image_side
        self.gt_type = gt_type
        assert (root / self.ground_truth / type).is_dir()
        if image_side == 'left':
            image_dir = root / self.image_left / type
        else:
            image_dir = root / self.image_right / type
        assert image_dir.is_dir()

        self.images = []
        self.annotations = []
        for root_dir, dirs, files in os.walk(image_dir):
            for file in files:
                self.images.append(Path(root_dir) / file)
                root_list = root_dir.split('/')
                annotations_path = root / self.ground_truth / root_list[-2] / root_list[-1]
                if image_side == 'right':
                    file_stem = re.split(r'/*_rightImg8bit.png', file)[0]
                else:
                    file_stem = re.split(r'/*_leftImg8bit.png', file)[0]
                self.annotations.append({
                    "polygons": annotations_path / (file_stem + "_" + self.ground_truth + "_polygons.json"),
                    "labelIds": annotations_path / (file_stem + "_" + self.ground_truth + "_labelIds.png"),
                    "instanceIds": annotations_path / (file_stem + "_" + self.ground_truth + "_instanceIds.png"),
                    "color": annotations_path / (file_stem + "_" + self.ground_truth + "_color.png"),
                })

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = pil_to_tensor(Image.open(self.images[idx]))
        annotation = pil_to_tensor(Image.open(self.annotations[idx][self.gt_type]))
        return image, annotation


if __name__ == "__main__":
    reader = CityScapes()
    print(len(reader))
    image, segmentation = reader[2]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation.permute(1, 2, 0))
    plt.show()
