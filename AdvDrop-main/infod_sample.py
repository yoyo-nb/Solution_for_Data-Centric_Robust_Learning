import numpy as np 
import json
import os
import sys
import time
import math
import io
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
# from torchvision import models  
import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
from  torchattacks.attack import Attack  
from utils import *
from compression import *
from decompression import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from models import *
import random
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_npy = '../data_3.npy'
label_npy = '../label_3.npy'
# arch = 'preactresnet18'
arch = 'wideresnet'

class InfoDrop(Attack):
    r"""    
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`. 
        
    """
    def __init__(self, model, height = 32, width = 32,  steps=40, batch_size = 20, block_size = 8, q_size = 10, targeted = False):
        super(InfoDrop, self).__init__("InfoDrop", model)
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Value for quantization range
        self.factor_range = [5, q_size]
        # Differential quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size) 
        q_ini_table = np.empty((batch_size,int(block_n),block_size,block_size), dtype = np.float32)
        q_ini_table.fill(q_size)
        self.q_tables = {"y": torch.from_numpy(q_ini_table),
                        "cb": torch.from_numpy(q_ini_table),
                        "cr": torch.from_numpy(q_ini_table)}        
    
     
    def forward(self, images, labels, labels_ind):
        r"""
        Overridden.
        """
        q_table = None
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        def cross_entropy(outputs, smooth_labels):
            loss = torch.nn.KLDivLoss(reduction='batchmean')
            return loss(F.log_softmax(outputs, dim=1), smooth_labels)
        # adv_loss =  nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([self.q_tables["y"],  self.q_tables["cb"], self.q_tables["cr"]], lr= 0.01)
        
        images = images.permute(0, 2, 3, 1)
        components = {'y': images[:,:,:,0], 'cb': images[:,:,:,1], 'cr': images[:,:,:,2]}
        for i in range(self.steps):
            self.q_tables["y"].requires_grad = True
            self.q_tables["cb"].requires_grad = True
            self.q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = block_splitting(components[k])
                # comp = 
                # import ipdb;ipdb.set_trace()
                comp = dct_8x8(comp)
                comp = quantize(comp, self.q_tables[k], self.alpha)
                comp = dequantize(comp, self.q_tables[k]) 
                comp = idct_8x8(comp)
                merge_comp = block_merging(comp, self.height, self.width)
                upresults[k] = merge_comp

            rgb_images = torch.cat([upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            # import ipdb;ipdb.set_trace()
            # print((rgb_images - images).max())
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            rgb_images_model = rgb_images / 255.0
            
            outputs = self.model(rgb_images_model)
            _, pre = torch.max(outputs.data, 1)
            if self.targeted:
                suc_rate = ((pre == labels_ind).sum()/self.batch_size).cpu().detach().numpy()
            else:
                suc_rate = ((pre != labels_ind).sum()/self.batch_size).cpu().detach().numpy()


            adv_cost = cross_entropy(outputs, labels)
            # print(labels.type(torch.long))
            # print(outputs)
            # adv_cost = adv_loss(outputs, labels.type(torch.long)) 
            
            if not self.targeted:
                adv_cost = -1* adv_cost

            total_cost = adv_cost 
            optimizer.zero_grad()
            total_cost.backward()

            self.alpha += self.alpha_interval
            
            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() -  torch.sign(self.q_tables[k].grad)
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()
            # if i%10 == 0:     
                # print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate )
            if suc_rate >= 1:
                # print('End at step {} with suc. rate {}'.format(i, suc_rate))
                q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
                return rgb_images, pre, i        
        q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
       
        return rgb_images, pre, q_table

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
        # import ipdb;ipdb.set_trace()
        # image = np.array(image)
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def save_img(img, img_name, save_dir):
    create_dir(save_dir)
    img_path = os.path.join(save_dir, img_name)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(img_path)
    
def pred_label_and_confidence(model, input_batch, labels_to_class):
    input_batch = input_batch.cuda()
    with torch.no_grad():
        out = model(input_batch)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1) * 100
    # print(percentage.shape)
    pred_list = []
    for i in range(index.shape[0]):
        pred_class = labels_to_class[index[i]]
        pred_conf =  str(round(percentage[i][index[i]].item(),2))
        pred_list.append([pred_class, pred_conf])
    return pred_list

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def load_model(arch):
    # import ipdb;ipdb.set_trace()
    model = globals()[arch]()
    ch = torch.load('../'+arch+'.pth.tar')
    model.load_state_dict(ch['state_dict'])
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),])  

    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    norm_layer = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # arch = 'preactresnet18'
    # arch = 'wideresnet'
    # arch = 'densenet121'
    name = 'advdrop_train_'+arch+'.npy'
    resnet_model = load_model(arch).cuda()
    resnet_model = nn.Sequential(
        norm_layer,
        resnet_model,
    ).to(device)
    resnet_model = resnet_model.eval()
    
    # Uncomment if you want save results
    save_dir = "./results"
    # create_dir(save_dir)
    batch_size = 1
    tar_cnt = 5000
    q_size = 40
    cur_cnt = 0
    suc_cnt = 0
    # data_dir = "./test-data"
    # data_clean(data_dir)
    # normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
    normal_data = MyDataset(transform)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)

    imgs = []
    normal_iter = iter(normal_loader)
    for i in range(tar_cnt//batch_size):
        
        images, labels = normal_iter.next()  
        # import ipdb;ipdb.set_trace()
        labels_ind = torch.argmax(labels, dim = 1).cuda()
        # For target attack: set random target. 
        # Comment if you set untargeted attack.
        # labels = torch.from_numpy(np.random.randint(0, 9, size = batch_size))
        
        images = images * 255.0
        attack = InfoDrop(resnet_model, batch_size=batch_size, q_size =q_size, steps=150, targeted = False)    

        # soft_label = np.zeros(10)
        # target = np.random.randint(0, 10)
        
        # while(target == labels_ind.tolist()[0]):
        #     target = np.random.randint(0, 10)
        # print(target, labels_ind.tolist()[0])
        # soft_label[target] += random.uniform(0, 10)
        # soft_label = torch.from_numpy(soft_label).cuda().type(labels.type()).unsqueeze(0)

        at_images, at_labels, suc_step = attack(images, labels, labels_ind)

        clip = lambda x: clip_tensor(x, 0, 1)
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        tf = transforms.Compose([
            # transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                # transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                transforms.Lambda(clip),
                                transforms.ToPILImage(),
                                # transforms.CenterCrop(224)
                                ])
        # Uncomment following codes if you wang to save the adv imgs
        # at_images_np = tf(at_images.cpu()) 
        # import ipdb;ipdb.set_trace()
        # at_images = at_images.astype(np.uint8)
        at_images /= 255.0
        for j in range(len(at_images)):
            img_p  = tf(at_images[j].cpu())
            img_p = np.array(img_p)
            # import ipdb;ipdb.set_trace()
            # img_p = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('js.jpg', img_p)
            # images[j] /= 255.0
            # img = np.array(tf(images[j]))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('j.jpg', img)
            # import ipdb;ipdb.set_trace()

            imgs.append(img_p)

        labels = labels.to(device)
        # print()
        suc_cnt += (at_labels != labels_ind).sum().item()
        # print()
        print("Iter: ", i, "Current suc. rate: ", suc_cnt/((i+1)*batch_size), at_labels.cpu().tolist(), labels_ind.cpu().tolist())
    images = np.array(imgs)
    np.save(name, images)
    
    # score_list = np.zeros(tar_cnt)
    # score_list[:suc_cnt] = 1.0
    # stderr_dist = np.std(np.array(score_list))/np.sqrt(len(score_list))
    # print('Avg suc rate: %.5f +/- %.5f'%(suc_cnt/tar_cnt,stderr_dist))
