import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import torchvision
import random
import cv2


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=3.0, amplitude=20.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255 # 避免有值超过255⽽反转
        img[img < 0] = 0
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

my_transform_gaussian_noise = AddGaussianNoise()
trans = transforms.RandomChoice([
    my_transform_gaussian_noise,
]) 

dataset = torchvision.datasets.CIFAR10(root = '../data', train=True, download=True)
images = []
soft_labels = []
for image, label in dataset:
    # im2 = np.array(image)
    # image = np.array(image)
    image = trans(image)
    image = np.array(image)
    # image = image[:,:,(2,1,0)]
    # im2 = im2[:,:,(2,1,0)]
    # import ipdb;ipdb.set_trace()
    # # img_p = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('js.jpg', image)
    # # im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('j.jpg', im2)
    
    # import ipdb;ipdb.set_trace()
    images.append(image)
    soft_label = np.zeros(10)
    soft_label[label] += random.uniform(0, 10) # an unnormalized soft label vector
    soft_labels.append(soft_label)
images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
np.save('data_naive.npy', images)
np.save('label_naive.npy', soft_labels)