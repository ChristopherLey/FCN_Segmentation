import os
import random
import shutil

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(
        self,
        path="../../../Datasets/carvana-image-masking-challenge",
        transform=None,
        validation=False,
    ):
        self.path = path
        self.transform = transform
        if validation:
            self.set = "val"
        else:
            self.set = "train"
        self.image_list = os.listdir(os.path.join(self.path, self.set))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.set, self.image_list[index])
        mask_path = os.path.join(
            self.path,
            f"{self.set}_masks",
            self.image_list[index].replace(".jpg", "_mask.gif"),
        )
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # preprocess mask (either 0 or 255) to be binary
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        return image, mask


def create_validation_set(path, set_size=48):
    assert os.path.exists(path), f"path:{path}, does not exist!"
    assert os.path.exists(f"{path}/train"), (
        f"Carvana test data dosen't exist in {path}/test\n"
        f"Please download the test data from "
        f"https://www.kaggle.com/c/carvana-image-masking-challenge/data"
    )
    if not os.path.exists(f"{path}/val"):
        os.mkdir(f"{path}/val")
        os.mkdir(f"{path}/val_masks")
    else:
        val_list = os.listdir(f"{path}/val")
        if len(val_list) // 16 == set_size:
            return
        else:
            for file in val_list:
                file_name = file[:-4]
                shutil.move(
                    f"{path}/val/{file_name}.jpg", f"{path}/train/{file_name}.jpg"
                )
                shutil.move(
                    f"{path}/val_masks/{file_name}_mask.gif",
                    f"{path}/train_masks/{file_name}_mask.gif",
                )
    uuid_list = []
    for file in os.listdir(f"{path}/train"):
        if file[:-7] not in uuid_list:
            uuid_list.append(file[:-7])
    for i in range(set_size):
        idx = random.randrange(0, len(uuid_list))
        uuid2move = uuid_list.pop(idx)
        for j in range(16):
            shutil.move(
                f"{path}/train/{uuid2move}_{j+1:02}.jpg",
                f"{path}/val/{uuid2move}_{j+1:02}.jpg",
            )
            shutil.move(
                f"{path}/train_masks/{uuid2move}_{j+1:02}_mask.gif",
                f"{path}/val_masks/{uuid2move}_{j+1:02}_mask.gif",
            )


if __name__ == "__main__":
    test_set = CarvanaDataset()
    for image, mask in test_set:
        print(image)
        break
