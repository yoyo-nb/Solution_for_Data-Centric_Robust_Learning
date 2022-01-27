import numpy as np
import random

img1 = np.load('data_1.npy')
img2 = np.load('naive/data_naive_2.npy')
img3 = np.load('AdvDrop-main/advdrop_train_wideresnet.npy')
img4 = np.load('AdvDrop-main/advdrop_train_preactresnet18.npy')
img5 = np.load('DeepFool/Python/deepfool_train_wideresnet.npy')
img6 =  np.load('DeepFool/Python/deepfool_train_preactresnet18.npy')
img7 = np.load('pgd/pdg_train_wideresnet.npy')  
img8 = np.load('pgd/pdg_train_preactresnet18.npy')
img9 = np.load('aa/aa_train_preactresnet18.npy')
img10 = np.load('aa/aa_train_wideresnet.npy')

la1 = np.load('label_1.npy')
la2 = np.load('label_2.npy')
la3 = np.load('label_3.npy')
la4 = np.load('label_4.npy')
la5 = np.load('label_5.npy')
la6 = np.load('label_6.npy')
la7 = np.load('label_7.npy')
la8 = np.load('label_8.npy')
la9 = np.load('label_9.npy')
la10 = np.load('label_10.npy')

labels = []
imgs = []
for i in range(5000):
    imgs.append(img1[i])
    labels.append(la1[i])

    imgs.append(img2[i+5000])
    labels.append(la2[i])

    imgs.append(img3[i])
    labels.append(la3[i])

    imgs.append(img4[i])
    labels.append(la4[i])

    imgs.append(img5[i])
    labels.append(la5[i])

    imgs.append(img6[i])
    labels.append(la6[i])

    imgs.append(img7[i])
    labels.append(la7[i])

    imgs.append(img8[i])
    labels.append(la8[i])

    imgs.append(img9[i])
    labels.append(la9[i])

    imgs.append(img10[i])
    labels.append(la10[i])

    
imgs = np.array(imgs)
labels = np.array(labels)
np.save('data.npy', imgs)
np.save('label.npy', labels)