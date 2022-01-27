import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
from models import *
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


data_npy = '../../data_6.npy'
label_npy = '../../label_6.npy'
# arch = 'wideresnet'
arch = 'preactresnet18'

# arch = 'densenet121'
name = 'deepfool_train_'+arch+'.npy'


def load_model(arch):
    model = globals()[arch]()
    ch = torch.load('../../'+arch+'.pth.tar')
    model.load_state_dict(ch['state_dict'])
    model.eval()
    return model


net = load_model(arch)

# Switch to evaluation mode
net.eval()


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def test():
    images = np.load(data_npy)
    labels = np.load(label_npy)
    images = [Image.fromarray(x) for x in images]
    labels = labels / labels.sum(axis=1, keepdims=True) # normalize
    labels = labels.astype(np.float32)

    # mean = [ 0.485, 0.456, 0.406 ]
    # std = [ 0.229, 0.224, 0.225 ]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Remove the mean
    im = transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                            std = std)])
    fool_img = []
    suc = 0
    for i in range(len(images)):
        inputs = images[i]
        soft_labels = labels[i]
        # inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        # im_orig = Image.open('test_im2.jpg')
        r, loop_i, label_orig, label_pert, pert_image = deepfool(im(inputs), net, overshoot=0.1)
        # print(label_orig==label_pert)
        if not label_orig==label_pert:
            suc +=1
        print(i+1, suc/(i+1))
        clip = lambda x: clip_tensor(x, 0, 255)

        tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                transforms.Lambda(clip),
                                transforms.ToPILImage(),
                                # transforms.CenterCrop(224)
                                ])

        img = (np.array(tf(pert_image.cpu()[0])))
        fool_img.append(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('t.jpg', img)
        # inputs.save('t1.jpg')
        # targets = soft_labels.argmax(dim=1)
        # outputs = model(inputs)
        # acc = accuracy(outputs, targets)
        # accs.update(acc[0].item(), inputs.size(0))
        # print(accs.avg)
        # import ipdb;ipdb.set_trace()
    # return accs.avg
    fool_img = np.array(fool_img)
    np.save(name, fool_img)

test()