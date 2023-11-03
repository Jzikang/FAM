from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import numpy as np
from random import choice
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--data', default='val', metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch_size', default=50, type=int, metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--size', default=299, type=int, metavar='N', help='the shape of resized image')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')

torch.backends.cudnn.benchmark = False
def main():
    global args
    feature = []
    # Data loading code
    args = parser.parse_args()
    """
    val_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(args.data, transforms.Compose([
                    transforms.ToTensor(),
                ])),
    batch_size=args.batchsize, shuffle=False,
    num_workers=args.workers, pin_memory=True)
    """
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    for i, (x, y) in enumerate(val_loader):
        print(i)
        front_20 = x[:20].cuda()
        feature.append(front_20.detach().cpu().numpy())
    feature = np.array(feature)
    np.save('./ImageNet_gallery.npy', feature)

if __name__ == "__main__":
    main()
