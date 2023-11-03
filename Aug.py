from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import random
import numpy as np
import os
import math
import argparse
from random import choice
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Normalize import Normalize
import cv2
import math
import timm
device = torch.device("cuda:0")
parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--data', default='val', metavar='DIR', help='path to dataset')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument('--eps', default=0.06, type=float, metavar='N', help='epsilon for attack perturbation')
parser.add_argument('--decay', default=1.0, type=float, metavar='N', help='decay for attack momentum')
parser.add_argument('--iteration', default=20, type=int, metavar='N', help='number of attack iteration')
parser.add_argument('-b', '--batchsize', default=20, type=int, metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--size', default=299, type=int, metavar='N', help='the size of image')
parser.add_argument('--resize', default=299, type=int, metavar='N', help='the resize of image')
parser.add_argument('--prob', default=0.5, type=float, metavar='N', help='probability of using diverse inputs.')
parser.add_argument('--num', '--data_num', default=20000, type=int, metavar='N', help='the num of test images')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',help='number of data loading workers (default: 4)')




def attack_mi(x, t_y, model):
    alpha = args.eps / args.iteration
    momentum = torch.zeros([args.batchsize, 3, args.size, args.size]).to(device)
    for i in range(args.iteration):
        pred_logit = model(x)
        real = pred_logit.gather(1,t_y.unsqueeze(1)).squeeze(1)
        ce_loss = -1 * real.sum()
        #ce_loss = F.cross_entropy(pred_logit.to(device), torch.tensor(t_y).to(device), reduction='sum').to(device) 
        print(ce_loss)
        ce_loss.backward()
        noise = x.grad.data
        l1_noise = torch.sum(torch.abs(noise), dim=(1, 2, 3))
        l1_noise = l1_noise[:, None, None, None]
        noise = noise / l1_noise
        momentum = momentum * args.decay + noise
        x = x - alpha * torch.sign(momentum)
        x = torch.clamp(x, 0, 1).detach()
        x.requires_grad = True
    return x




def save_img(save_path, img):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = (img * 255).permute(0, 2, 3, 1).detach().cpu()
    # print(img.shape[0])
    for i in range(img.shape[0]):
        img_name = os.path.join(save_path, str(i) + '.png')
        Image.fromarray(np.array(img[i].squeeze(0)).astype('uint8')).save(img_name)




def main():
    global args
    args = parser.parse_args()
    val_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(args.data, transforms.Compose([
                    transforms.Resize(320),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                ])),
    batch_size=args.batchsize, shuffle=False,
    num_workers=args.workers, pin_memory=True)
    densenet_121 = torch.nn.Sequential(Normalize(args.mean, args.std), models.densenet121(pretrained=True).eval().to(device))
    resnet_18 = torch.nn.Sequential(Normalize(args.mean, args.std), models.resnet18(pretrained=True).eval().to(device))
    vgg_11 = torch.nn.Sequential(Normalize(args.mean, args.std), models.vgg11_bn(pretrained=True).eval().to(device))

    for i, (x, y) in (enumerate(val_loader)):
        if i != args.num // args.batchsize:
            x = Variable(x.to(device), requires_grad=True)
            target_y = torch.tensor([i]*args.batchsize).to(device)
            x_aug = attack_mi(x,target_y,vgg_11)
                
            root = "./vgg_aug"
            save_path = os.path.join(root, str(i))
            save_img(save_path, x_aug)     
        else:
            break
    file_list = os.listdir(root)
 
    for file in file_list:
        filename = file.zfill(4)
        new_name = ''.join(filename)
        os.rename(root + '/' + file, root + '/' + new_name)

if __name__ == "__main__":
    main()