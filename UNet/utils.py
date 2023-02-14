import os
from datetime import datetime

import torch
import torchvision
from carvana_dataset_tools import CarvanaDataset
from torch.utils.data import DataLoader


class eval(object):
    def __init__(self, model=None):
        self.model = model

    def __enter__(self):
        if self.model is not None:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            self.model.train()


def load_checkpoint(model, checkpoint):
    print("==> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def save_checkpoint(state, filename=None, epoch=None):
    if filename is None:
        now = datetime.now()
        if epoch is not None:
            filename = f"./checkpoints/checkpoint_{now.strftime('%Y-%m-%d_%H:%M')}_epoch_{epoch}.pth.tar"
        else:
            filename = (
                f"./checkpoints/checkpoint_{now.strftime('%Y-%m-%d_%H:%M')}.pth.tar"
            )

    print(f"==> Saving Checkpoint to: {filename}")
    torch.save(state, filename)


def get_loaders(
    path,
    batch_size,
    train_transform,
    val_transforms,
    num_workers=6,
    pin_memory=True,
    dataset="carvana",
):
    if dataset == "carvana":
        train_dataset = CarvanaDataset(
            path, transform=train_transform, validation=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        validation_dataset = CarvanaDataset(
            path, transform=val_transforms, validation=True
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    else:
        raise Exception(f"dataset={dataset}, is not a valid dataset!")

    return train_loader, validation_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with eval(model):
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                predictions = torch.sigmoid(model(x))
                # TODO: adapt for multiclass (only binary atm)
                predictions = (predictions > 0.5).float()
                num_correct += (predictions == y).sum()
                num_pixels += torch.numel(predictions)
                # Dice score is a simplification of intersection of union for binary classification
                dice_score += (2 * (predictions * y).sum()) / (
                    (predictions + y).sum() + 1e-8
                )
                # TODO: add Interection of Union for multiclass

    print(
        f"Accuracy: {num_correct}/{num_pixels} = {num_correct/num_pixels*100.0:.2f}%\n"
        f"Dice score: {dice_score/len(loader)}"
    )


def save_predictions_as_images(loader, model, path="./saved_images", device="cuda"):
    with eval(model):
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                prediction = torch.sigmoid(model(x))
                prediction = (prediction > 0.5).float()
            torchvision.utils.save_image(
                prediction, os.path.join(path, f"prediction_{idx}.png")
            )
            torchvision.utils.save_image(
                y.unsqueeze(1), os.path.join(path, f"ground_truth_{idx}.png")
            )
