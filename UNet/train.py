import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from model import UNet
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from utils import check_accuracy
from utils import get_loaders
from utils import load_checkpoint
from utils import save_checkpoint
from utils import save_predictions_as_images

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_WORKERS = 6
IMAGE_HEIGHT = 1280 // 2  # 1280 originally
IMAGE_WIDTH = 1904 // 2  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
PATH = "../../../Datasets/carvana-image-masking-challenge"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)  # nice progress bar

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    loss_fn = (
        nn.BCEWithLogitsLoss()
    )  # Binary Cross-Entropy logits because not sigmoid on output of model,
    # change to cross entropy loss for multichannel

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        PATH,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        dataset="carvana",
    )

    if LOAD_MODEL:
        load_checkpoint(
            model,
            torch.load("./checkpoints/checkpoint_2022-01-06_13:12_epoch_4.pth.tar"),
        )

    check_accuracy(val_loader, model, device=device)
    scaler = GradScaler()  # float16 scaling of gradient

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, epoch=epoch)

        # check accuracy
        check_accuracy(val_loader, model, device=device)

        # print some examples to a folder
        save_predictions_as_images(
            val_loader, model, path="./saved_images", device=device
        )


if __name__ == "__main__":
    main()
