import os
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image

from datasets.settings import *
from datasets.transforms import train_transform
from datasets.transforms import valid_transform


class IMAGENET1K:
    def __init__(self, root, train=True, transform=None):
        train_txt_path = os.path.join(root, "train.txt")
        valid_txt_path = os.path.join(root, "valid.txt")
        if not os.path.isfile(train_txt_path) or not os.path.isfile(valid_txt_path):
            raise Exception("Please generate train.txt and valid.txt first.")
        train_txt = open(train_txt_path, 'r', encoding='utf-8')
        valid_txt = open(valid_txt_path, 'r', encoding='utf-8')

        data_txt = train_txt if train else valid_txt
        split_folder = "train" if train else "valid"
        data = data_txt.readlines()
        img_paths = [os.path.join(root, split_folder, dat.split()[0]) for dat in data]
        labels = [int(dat.split()[1]) for dat in data]

        self.root = root
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

        assert len(self.img_paths) == len(self.labels), "ERROR: CANNOT GETTING DATA!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        image = Image.open(img_path)
        if image.layers == 1:
          image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return torch.Tensor(image), label


ROOT = os.path.join(DATA_DIR, "ILSVRC2012")
train_data = IMAGENET1K(root=ROOT, train=True, transform=train_transform)
valid_data = IMAGENET1K(root=ROOT, train=False, transform=valid_transform)
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2)
