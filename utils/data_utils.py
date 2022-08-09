import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from skimage.io import imread
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import os

image_size = (224, 224)
num_classes = 2
epochs = 150
num_workers = 4
img_data_dir = '/vol/biomedic3/mi615/datasets/SkinLesion/ISIC_ARCHIVE/Images_224/'


class SkinDataset(Dataset):
    def __init__(self, csv_file_img, augmentation=False):
        self.data = pd.read_csv(csv_file_img)
        self.do_augment = augmentation

        self.labels = [
            'benign',
            'malignan']

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.data['images'] = img_data_dir + self.data['images']
        img_label = (self.data['benign_malignan'] == 'malignan').astype(int)
        self.samples = {'image_path': list(self.data['images']) , 'label': list(img_label)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).permute(2,0,1)
        label = torch.tensor(sample['label'])

        if self.do_augment:
            image = self.augment(image)

        return image, label#{'image': image, 'label': label}

    def get_sample(self, item):
        image = imread(self.samples['image_path'][item]).astype(np.float32)
        return {'image': image, 'label': self.samples['label'][item]}



def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    
    csv_train_img = '../datafiles/isic_train.csv'
    csv_val_img = '../datafiles/isic_test.csv'
    trainset = SkinDataset(csv_train_img, augmentation=True)
    testset = SkinDataset(csv_val_img, augmentation=False)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
