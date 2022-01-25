import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import namedtuple
from torch.utils.data import DataLoader
from CityScapes.tools.data import CityScapesFineDataset
from datetime import datetime
import torch
from torchmetrics import JaccardIndex
import torchvision
import os


class eval(object):
    def __init__(self, model=None):
        self.model = model

    def __enter__(self):
        if self.model is not None:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            self.model.train()


def load_checkpoint(checkpoint_dir):
    print("==> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_dir)
    return checkpoint


def save_checkpoint(checkpoint, filename=None, epoch=None):
    if filename is None:
        now = datetime.now()
        if epoch is not None:
            filename = f"./checkpoints/checkpoint_{now.strftime('%Y-%m-%d_%H:%M')}_epoch_{epoch}.pth.tar"
        else:
            filename = f"./checkpoints/checkpoint_{now.strftime('%Y-%m-%d_%H:%M')}.pth.tar"

    print(f"==> Saving Checkpoint to: {filename}")
    torch.save(checkpoint, filename)


def check_accuracy(
        loader,
        model,
        device="cuda",
        num_classes=34,
        absent_score=0.0,
        threshold=0.5,
        reduction='elementwise_mean'
):
    score_sum = 0
    count = 0
    jaccard_score = JaccardIndex(
        num_classes=num_classes,
        absent_score=absent_score,
        threshold=threshold,
        reduction=reduction
    )
    with eval(model):
        with torch.no_grad():
            for image, mask in loader:
                count += 1
                image = image.to(device)
                mask = mask.to(device).unsqueeze(1)
                predictions = torch.sigmoid(model(image))
                score = jaccard_score(predictions, mask)
                score_sum += score

    print(
        f"Accuracy: average_score {score_sum/count:.2f}\n"
    )


def save_predictions_as_images(
        loader, model, path="./saved_images", device="cuda"
):
    with eval(model):
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                prediction = torch.sigmoid(model(x))
                prediction = (prediction > 0.5).float()
            torchvision.utils.save_image(
                prediction, os.path.join(path, f'prediction_{idx}.png')
            )
            torchvision.utils.save_image(
                y.unsqueeze(1), os.path.join(path, f'ground_truth_{idx}.png')
            )


def get_loaders(
        path,
        batch_size,
        train_transform,
        val_transform,
        num_workers=6,
        pin_memory=True,
):
    train_dataset = CityScapesFineDataset(
        path,
        dataset="train",
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    validation_dataset = CityScapesFineDataset(
        path,
        dataset='val',
        transform=val_transform
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    return train_loader, validation_loader


def show_example_images(file_set, file, directory="/home/chris/Dropbox/AI/Datasets/CityScapes"):
    file_meta = file.split('_')
    mask_dir = f"{directory}/gtFine_trainvaltest/gtFine/{file_set}/{file_meta[0]}"
    left_image_dir = f"{directory}/leftImg8bit_trainvaltest/leftImg8bit/{file_set}/{file_meta[0]}"
    right_image_dir = f"{directory}/rightImg8bit_trainvaltest/rightImg8bit/{file_set}/{file_meta[0]}"
    im_label_ids = Image.open(f'{mask_dir}/{file}_gtFine_labelIds.png')
    im_colour = Image.open(f'{mask_dir}/{file}_gtFine_color.png')
    left_image = Image.open(f'{left_image_dir}/{file}_leftImg8bit.png').convert('RGB')
    right_image = Image.open(f'{right_image_dir}/{file}_rightImg8bit.png').convert('RGB')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.set_title("Left Camera")
    ax1.imshow(left_image)

    ax2.set_title("Right Camera")
    ax2.imshow(right_image)

    ax3.set_title("Label Ids")
    ax3.imshow(im_label_ids)

    ax4.set_title("Colour Map")
    ax4.imshow(im_colour)
    plt.show()

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


if __name__ == "__main__":
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format('name', 'id', 'trainId', 'category',
                                                                                  'categoryId', 'hasInstances',
                                                                                  'ignoreInEval'))
    print("    " + ('-' * 98))
    for label in labels:
        print(
            "    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format(label.name, label.id, label.trainId,
                                                                                    label.category, label.categoryId,
                                                                                    label.hasInstances,
                                                                                    label.ignoreInEval))
    print("")
    show_example_images("train", "erfurt_000000_000019")