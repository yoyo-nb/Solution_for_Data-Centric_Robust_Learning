import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from models import *
import os
import cv2
from PIL import Image
import random
import torch.utils.data as data
import torchattacks
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_npy = '../data_10.npy'
label_npy = '../label_10.npy'
# arch = 'preactresnet18'
arch = 'wideresnet'


# arch = 'densenet121'
name = 'aa_train_'+arch+'.npy'
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def load_model(arch):
    model = globals()[arch]()
    ch = torch.load('../'+arch+'.pth.tar')
    model.load_state_dict(ch['state_dict'])
    model.eval()
    return model

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load(data_npy)
        labels = np.load(label_npy)
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean,std),
])
trainset = MyDataset(transform_train)
trainloader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)

model = load_model(arch)
model = nn.Sequential(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        model,
        # models.resnet50(pretrained=True)
).to(device)
clip = lambda x: clip_tensor(x, 0, 1)
tf = transforms.Compose([
    transforms.Lambda(clip),
    transforms.ToPILImage(),
    # transforms.CenterCrop(224)
])

model.eval()


attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='plus', n_classes=10, seed=None, verbose=False)
num =0

suc = 0
imgs = []
for data, target in trainloader:
    num += 1
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    tar_ind = target.max(1, keepdim=True)[1].cpu()

    soft_label = np.zeros(10)
    new_target = np.random.randint(0, 10)
    
    adv_image = attack(data, tar_ind[0])
    # Forward pass the data through the model
    output = model(adv_image)
    init_pred = output.max(1, keepdim=True)[1].cpu() # get the index of the max log-probability

    img_p  = tf(adv_image[0].cpu())
    img_p = np.array(img_p)
    imgs.append(img_p)
    if tar_ind != init_pred:
        suc+=1
    print(num, suc/num, tar_ind, init_pred)


    # img_p = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('js.jpg', img_p)
    # img_p  = tf(data[0].cpu())
    # img_p = np.array(img_p)
    # img_p = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('j.jpg', img_p)
    # import ipdb;ipdb.set_trace()

imgs = np.array(imgs)
np.save(name, imgs)


