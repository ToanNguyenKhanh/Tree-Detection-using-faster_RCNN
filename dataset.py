import os
import cv2
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pprint import pprint
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, Normalize
from PIL import Image


class TreeDataset(Dataset):
    def __init__(self, root='Tree detection', train=True, transform=None):
        self.classes = ['Tree']
        self.image_paths = []
        self.image_names = []
        self.labels = []
        if train:
            data_paths = os.path.join(root, 'train')
            self.annotation_path = os.path.join(data_paths, '_annotations.coco.json')
        else:
            data_paths = os.path.join(root, 'valid')
            self.annotation_path = os.path.join(data_paths, '_annotations.coco.json')

        with open(self.annotation_path, "r") as json_file:
            json_data = json.load(json_file)
            annotations = json_data['annotations']
            images_json = json_data['images']

        for img in os.listdir(data_paths):
            image_name = img
            has_annotation = False
            for im in images_json:
                if image_name == im['file_name']:
                    has_annotation = True
                    break
            if has_annotation:
                img_path = os.path.join(data_paths, img)
                self.image_names.append(img)
                self.image_paths.append(img_path)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, item):
        boxes, labels = [], []

        image = cv2.imread(self.image_paths[item])
        image_name = self.image_names[item]

        with open(self.annotation_path, "r") as json_file:
            json_data = json.load(json_file)
            annotations = json_data['annotations']
            images_json = json_data['images']

        id = None
        for img in images_json:
            if image_name == img['file_name']:
                id = int(img['id'])
                break

        for ann in annotations:
            if ann['image_id'] == id:
                bbox = ann['bbox']
                xmin, ymin, width, height = map(int, bbox)
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])

        target = {
            "boxes": torch.FloatTensor(boxes),
            "labels": torch.LongTensor(labels)
        }
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, target


if __name__ == '__main__':
    training_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    datasets = TreeDataset(train=True, transform=training_transform)
    print(datasets.__len__())
    images, targets = datasets[1052]
    print(targets)



