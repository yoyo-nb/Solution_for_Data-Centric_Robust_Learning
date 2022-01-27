import numpy as np
import torchvision
import random

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
images = []
soft_labels = []
ind = 0
for image, label in dataset:
    image = np.array(image)
    images.append(image)
    soft_label = np.zeros(10)
    soft_label[label] += random.uniform(0, 10) # an unnormalized soft label vector
    soft_labels.append(soft_label)
    ind += 1
    if(ind % 5000 == 0):
        images = np.array(images)
        soft_labels = np.array(soft_labels)
        print(soft_labels.sum(0))
        print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
        np.save(f'data_{ind//5000}.npy', images)
        np.save(f'label_{ind//5000}.npy', soft_labels)
        images = []
        soft_labels = []

