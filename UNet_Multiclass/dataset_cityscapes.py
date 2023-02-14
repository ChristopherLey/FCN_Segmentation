import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

# fmt: off
Label = namedtuple(
    "Label",
    [
        "name",             # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
        "id",               # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.
        "trainId",          # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
                            # For trainIds, multiple labels might have the same ID. Then, these labels
                            # are mapped to the same class in the ground truth images. For the inverse
                            # mapping, we use the label that is defined first in the list below.
                            # For example, mapping all void-type classes to the same ID in training,
                            # might make sense for some approaches.
                            # Max value is 255!
        "category",         # The name of the category that this label belongs to
        "categoryId",       # The ID of this category. Used to create ground truth images
                            # on category level.
        "hasInstances",     # Whether this label distinguishes between single instances or not
        "ignoreInEval",     # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not
        "color",            # The color of this label
    ],
)
#
labels = [
    #       name                    id  trainId     category    catId   hasInstances    ignoreInEval    color
    Label("unlabeled",              0,  255,        "void",         0,      False,          True,       (0, 0, 0)),     # noqa: E241,E501
    Label("ego vehicle",            1,  255,        "void",         0,      False,          True,       (0, 0, 0)),     # noqa: E241,E501
    Label("rectification border",   2,  255,        "void",         0,      False,          True,       (0, 0, 0)),     # noqa: E241,E501
    Label("out of roi",             3,  255,        "void",         0,      False,          True,       (0, 0, 0)),     # noqa: E241,E501
    Label("static",                 4,  255,        "void",         0,      False,          True,       (0, 0, 0)),     # noqa: E241,E501
    Label("dynamic",                5,  255,        "void",         0,      False,          True,       (111, 74, 0)),  # noqa: E241,E501
    Label("ground",                 6,  255,        "void",         0,      False,          True,       (81, 0, 81)),   # noqa: E241,E501
    Label("road",                   7,  0,          "flat",         1,      False,          False,      (128, 64, 128)),    # noqa: E241,E501
    Label("sidewalk",               8,  1,          "flat",         1,      False,          False,      (244, 35, 232)),    # noqa: E241,E501
    Label("parking",                9,  255,        "flat",         1,      False,          True,       (250, 170, 160)),   # noqa: E241,E501
    Label("rail track",             10, 255,        "flat",         1,      False,          True,       (230, 150, 140)),   # noqa: E241,E501
    Label("building",               11, 2,          "construction", 2,      False,          False,      (70, 70, 70)),  # noqa: E241,E501
    Label("wall",                   12, 3,          "construction", 2,      False,          False,      (102, 102, 156)),   # noqa: E241,E501
    Label("fence",                  13, 4,          "construction", 2,      False,          False,      (190, 153, 153)),   # noqa: E241,E501
    Label("guard rail",             14, 255,        "construction", 2,      False,          True,       (180, 165, 180)),   # noqa: E241,E501
    Label("bridge",                 15, 255,        "construction", 2,      False,          True,       (150, 100, 100)),   # noqa: E241,E501
    Label("tunnel",                 16, 255,        "construction", 2,      False,          True,       (150, 120, 90)),    # noqa: E241,E501
    Label("pole",                   17, 5,          "object",       3,      False,          False,      (153, 153, 153)),   # noqa: E241,E501
    Label("polegroup",              18, 255,        "object",       3,      False,          True,       (153, 153, 153)),   # noqa: E241,E501
    Label("traffic light",          19, 6,          "object",       3,      False,          False,      (250, 170, 30)),    # noqa: E241,E501
    Label("traffic sign",           20, 7,          "object",       3,      False,          False,      (220, 220, 0)),     # noqa: E241,E501
    Label("vegetation",             21, 8,          "nature",       4,      False,          False,      (107, 142, 35)),    # noqa: E241,E501
    Label("terrain",                22, 9,          "nature",       4,      False,          False,      (152, 251, 152)),   # noqa: E241,E501
    Label("sky",                    23, 10,         "sky",          5,      False,          False,      (70, 130, 180)),    # noqa: E241,E501
    Label("person",                 24, 11,         "human",        6,      True,           False,      (220, 20, 60)),     # noqa: E241,E501
    Label("rider",                  25, 12,         "human",        6,      True,           False,      (255, 0, 0)),   # noqa: E241,E501
    Label("car",                    26, 13,         "vehicle",      7,      True,           False,      (0, 0, 142)),   # noqa: E241,E501
    Label("truck",                  27, 14,         "vehicle",      7,      True,           False,      (0, 0, 70)),    # noqa: E241,E501
    Label("bus",                    28, 15,         "vehicle",      7,      True,           False,      (0, 60, 100)),  # noqa: E241,E501
    Label("caravan",                29, 255,        "vehicle",      7,      True,           True,       (0, 0, 90)),    # noqa: E241,E501
    Label("trailer",                30, 255,        "vehicle",      7,      True,           True,       (0, 0, 110)),   # noqa: E241,E501
    Label("train",                  31, 16,         "vehicle",      7,      True,           False,      (0, 80, 100)),  # noqa: E241,E501
    Label("motorcycle",             32, 17,         "vehicle",      7,      True,           False,      (0, 0, 230)),   # noqa: E241,E501
    Label("bicycle",                33, 18,         "vehicle",      7,      True,           False,      (119, 11, 32)),     # noqa: E241,E501
    Label("license plate",          -1, -1,         "vehicle",      7,      False,          True,       (0, 0, 142)),   # noqa: E241,E501
]
# fmt: on


class CityScapes(Dataset):
    ground_truth = "gtFine"
    image_left = "leftImg8bit"
    image_right = "rightImg8bit"

    def __init__(
        self,
        root: Path = Path("/data/Datastore/Cityscapes"),
        type: str = "train",
        image_side: str = "left",
        gt_type: str = "labelIds",
    ):
        assert type in ["train", "test", "val"]
        assert image_side in ["left", "right"]
        assert gt_type in ["labelIds", "polygons", "instanceIds", "color"]
        self.image_side = image_side
        self.gt_type = gt_type
        assert (root / self.ground_truth / type).is_dir()
        if image_side == "left":
            image_dir = root / self.image_left / type
        else:
            image_dir = root / self.image_right / type
        assert image_dir.is_dir()
        self.labels = labels
        self.images = []
        self.annotations = []
        for root_dir, dirs, files in os.walk(image_dir):
            for file in files:
                self.images.append(Path(root_dir) / file)
                root_list = root_dir.split("/")
                annotations_path = (
                    root / self.ground_truth / root_list[-2] / root_list[-1]
                )
                if image_side == "right":
                    file_stem = re.split(r"/*_rightImg8bit.png", file)[0]
                else:
                    file_stem = re.split(r"/*_leftImg8bit.png", file)[0]
                self.annotations.append(
                    {
                        "polygons": annotations_path
                        / (file_stem + "_" + self.ground_truth + "_polygons.json"),
                        "labelIds": annotations_path
                        / (file_stem + "_" + self.ground_truth + "_labelIds.png"),
                        "instanceIds": annotations_path
                        / (file_stem + "_" + self.ground_truth + "_instanceIds.png"),
                        "color": annotations_path
                        / (file_stem + "_" + self.ground_truth + "_color.png"),
                    }
                )
        self.num_classes = 33

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
