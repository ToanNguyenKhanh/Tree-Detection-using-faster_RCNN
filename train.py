import os
import argparse
from dataset import TreeDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, Normalize
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='Tree detection')
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_path', type=str, default='trained_models')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='trained_models/last.pt')
    parser.add_argument('-t', '--tensorboard_path', type=str, default='tensorboard')
    args = parser.parse_args()
    return args

def collate(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train(args):

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, trainable_backbone_layers=5).to(device)

    # Define data transformations for training and validation datasets
    training_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    # Create training and validation datasets
    trainset = TreeDataset(args.data_path, train=True, transform=training_transform)
    valset = TreeDataset(args.data_path, train=False, transform=val_transform)

    # Create training and validation dataloaders
    training_dataloader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True,
        collate_fn=collate
    )

    val_dataloader = DataLoader(
        dataset=valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        collate_fn=collate
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # load if model existed
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_map = checkpoint['best_map']
    else:
        start_epoch = 0
        best_map = 0

    # Create a checkpoint directory for training
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # Create a directory for Tensorboard logs
    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)

    # Set up Tensorboard writer
    writer = SummaryWriter(args.tensorboard_path)
    num_iters = len(training_dataloader)

    #  Loop through each epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(training_dataloader, colour='cyan')

        # Loop through each batch
        for i, (images, targets) in enumerate(progress_bar):
            # put images, targets to device
            images = [image.to(device) for image in images]
            list_targets = []
            for target in targets:
               list_targets.append({key: value.to(device) for key, value in target.items()})

            loss_components = model(images, list_targets)
            losses = sum(loss for loss in loss_components.values())
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, args.epochs, losses))
            writer.add_scalar("Train/loss", losses, epoch * len(training_dataloader) + i)
            # optimize
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        model.eval()
        metric = MeanAveragePrecision(iou_type='bbox')
        progress_bar = tqdm(val_dataloader, colour='green')
        for i, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                predictions = model(images)
            list_targets = []
            for target in targets:
                list_targets.append({key: value.to(device) for key, value in target.items()})
            metric.update(predictions, list_targets)

        map = metric.compute()
        writer.add_scalar("Val/mAP", map["map"], epoch)
        writer.add_scalar("Val/mAP50", map["map_50"], epoch)
        writer.add_scalar("Val/mAP75", map["map_75"], epoch)

        checkpoint = {
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch' : epoch+1,
            'best_map': map["map"]
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'last.pt'))
        if map["map"] > best_map:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_map = map["map"]

if __name__ == '__main__':
    args = parse_args()
    train(args)
