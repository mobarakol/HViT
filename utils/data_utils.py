import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Sampler

from skimage.io import imread
import torchvision.transforms as T
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
import pickle
import random
from PIL import Image, ImageOps

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // 3
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self,degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // 3
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // 3
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img,brightness_factor)
        img_ = TF.adjust_contrast(img_,contrast_factor)
        img_ = TF.adjust_saturation(img_,saturation_factor)
        img_ = TF.adjust_hue(img_,hue_factor)
        
        return img_




class CholecDataset(Dataset):
    ''' Dataset class for Grad-CAM model
        input: images directory, intruments annotations, transform configurations, image loader
        output: image and label
    '''
    
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_tool = file_labels[:,0]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_tool = self.file_labels_tool[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_tool

    def __len__(self):
        return len(self.file_paths)

def get_data(data_path):
    ''' prepare the data for dataloader
        input: pickle file containing data directory and labels
        output: training dataset, number of train images in each sequence, validation dataset, number of val images in each sequences
    '''

    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_40 = train_test_paths_labels[0]
    val_paths_40 = train_test_paths_labels[1]
    train_labels_40 = train_test_paths_labels[2]
    val_labels_40 = train_test_paths_labels[3]
    train_num_each_40 = train_test_paths_labels[4]
    val_num_each_40 = train_test_paths_labels[5]

    # print('train_paths_40  : {:6d}'.format(len(train_paths_40)))
    # print('train_labels_40 : {:6d}'.format(len(train_labels_40)))
    # print('valid_paths_40  : {:6d}'.format(len(val_paths_40)))
    # print('valid_labels_40 : {:6d}'.format(len(val_labels_40)))

    train_labels_40 = np.asarray(train_labels_40, dtype=np.int64)
    val_labels_40 = np.asarray(val_labels_40, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.4084945, 0.25513682, 0.25353566], [0.22662906, 0.20201652, 0.1962526 ])
        ])

    test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4084945, 0.25513682, 0.25353566], [0.22662906, 0.20201652, 0.1962526 ])
        ])
    
    train_dataset_40 = CholecDataset(train_paths_40, train_labels_40, train_transforms)
    val_dataset_40 = CholecDataset(val_paths_40, val_labels_40, test_transforms)

    return train_dataset_40, train_num_each_40, \
           val_dataset_40, val_num_each_40


class SeqSampler(Sampler):
    ''' sample the data for dataloader according to the index
        input: data source, index of all frames in every sequence set 
    '''

    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    
    pkl_cholec40 = '../../CholecT80Classification/train_val_paths_labels_adjusted.pkl'
    pkl_endovis18 = '../../CholecT80Classification/miccai2018_train_val_paths_labels_adjusted.pkl'
    train_dataset_40, train_num_each_40, \
    val_dataset_40, val_num_each_40 = get_data(pkl_cholec40)
    train_dataset_18, train_num_each_18, \
    val_dataset_18, val_num_each_18 = get_data(pkl_endovis18)

    trainset = train_dataset_40 + train_dataset_18
    train_num_each = train_num_each_40 + train_num_each_18

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(val_dataset_40)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(val_dataset_40,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if val_dataset_40 is not None else None

    return train_loader, test_loader
