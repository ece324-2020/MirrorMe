# Want to load:
# - Target image
# - Source image embedding
# = Target image embedding

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torchvision
import os
from PIL import Image

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

# IMAGE_PATH = ""
# transform = None
# dataset = ImageFolder(IMAGE_PATH, transform)

class GAN_Dataset(data.Dataset):

    def __init__(self, dir, transform):
        super(GAN_Dataset, self).__init__()
        self.dir = dir
        self.transform = transform
        self.total_imgs = os.listdir(dir)

        self.img_paths = []
        img_path = self.dir + "/"
        img_list = os.listdir(dir)
        img_nums = len(img_list)
        for i in range(img_nums):
            img_name = img_path + img_list[i]
            self.img_paths.append(img_name)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)

        name = self.img_paths[idx]
        print(name)
        return tensor_image

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def dataloader(root_source = "\clean_dataset\train_data",
               root_target = "\clean_dataset\train_data",
               image_size = 224,
               num_channels = 3,
               batch_size = 4,
               num_workers = 6,
               shuffle = True):

    transform = transforms.Compose([
        transforms.ToTensor(),
#        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        transforms.Resize((image_size,image_size)),
#        transforms.RandomRotation(45),
    ])

    # image_data = GAN_Dataset(dir = root, transform = transform)
    # dataset = data.TensorDataset(image_data, image_data)
    # dataloader = data.DataLoader(image_data,
    #                                batch_size=batch_size,
    #                                shuffle=True,
    #                                num_workers=num_workers)
    
    # return zip(dataloader, dataloader)

    dataloader = data.DataLoader(
    ConcatDataset(
        GAN_Dataset(dir = root_source, transform = transform), 
        GAN_Dataset(dir = root_target, transform = transform)
    ), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
