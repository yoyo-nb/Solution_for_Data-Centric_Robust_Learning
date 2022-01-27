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
from models import *
import os
import cv2
from PIL import Image
import random
import torch.utils.data as data
import torchattacks
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_npy = '../data_8.npy'
label_npy = '../label_8.npy'
arch = 'preactresnet18'
# arch = 'wideresnet'


# arch = 'densenet121'
name = 'pdg_train_'+arch+'.npy'
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
def pgd_attack(model, images, labels, eps=0.25, alpha=2/255, iters=40) :
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    tar_ind = labels.max(1, keepdim=True)[1].cpu()[0][0]
    loss = nn.CrossEntropyLoss()
        
    adv_images = images.clone().detach()
    ### random start
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    # ori_images = images.data
    
    soft_label = np.zeros(10)
    target = np.random.randint(0, 10)
    
    while(target == tar_ind.tolist()):
        target = np.random.randint(0, 10)
    print(target, tar_ind.tolist())
    soft_label[target] += random.uniform(0, 10)
    soft_label = torch.from_numpy(soft_label).cuda().type(labels.type()).unsqueeze(0)

    for i in range(iters) :    
        adv_images.requires_grad = True
        outputs = model(adv_images)

        model.zero_grad()
        
        cost = -cross_entropy(outputs, soft_label)
        # cost = cross_entropy(outputs, labels)
        # cost.backward()

        grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
    return adv_images
# attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)

num =0
suc = 0
imgs = []
for data, target in trainloader:
    num += 1
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    tar_ind = target.max(1, keepdim=True)[1].cpu()
    # Set requires_grad attribute of tensor. Important for Attack
    # data.requires_grad = True
    # adv_images = attack(data, target)
    adv_image = pgd_attack(model, data, target)
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


