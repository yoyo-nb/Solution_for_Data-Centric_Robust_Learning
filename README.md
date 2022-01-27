

Top3 solution code for [AAAI2022 安全AI挑战者计划第八期：Data-Centric Robust Learning on ML Models](https://tianchi.aliyun.com/competition/entrance/531939/introduction).

# Solution Dataset 
- 5k images: Original cifar10 training set.
- 5k images: Original cifar10 training set with Gaussian noise.
- 10k images: Adversarial examples of the cifar10 training set attacked by AdvDrop. (5k for `preactresnet18` and 5k for `wideresnet`. Similar to the following.)
- 10k images: Adversarial examples of the cifar10 training set attacked by DeepFool.
- 10k images: Adversarial examples of the cifar10 training set attacked by PGD.
- 10k images: Adversarial examples of the cifar10 training set attacked by AutoAttack.

Total: 50k images.

# Generate the Solution Dataset

## 0. Environment
```
pip install -r requirements.txt
```

## 1. Train the two model using the original cifar10 training set.

- Change code at line 5 in `gen_dataset.py`: `train=True`.
- Get the original cifar10 training set `data.npy` and `label.npy`:
    ```
    python gen_dataset.py
    ```
- Change code at lines 14 and 29 in `config.py`: `'batch_size': 64`
- Train the two model, get the trained model `preactresnet18.pth.tar` and `wideresnet.pth.tar`:
    ```
    python train.py
    ```

## 2. Split the original cifar10 training set into 10 parts.

```
python split.py
```

The training set will be split into 10 parts: `data_1.npy`, `label_1.npy`, `data_2.npy`, `label_2.npy`, ... , `data_10.npy`, `label_10.npy`.

Each part has 5k images.

## 3. Get each part of the Solution Dataset

### 3.1 Gaussian noise for `data_2.npy`

```
cd naive
python naive.py
```
The data will be saved in `data_naive.npy` and `label_naive.npy`

### 3.2 AdvDrop for `data_3.npy` and `data_4.npy`
- Change codes at lines 26-29 in 'AdvDrop-main/infod_sample.py'
    ```
    data_npy = '../data_3.npy'
    label_npy = '../label_3.npy'
    arch = 'wideresnet'
    ```
- Run and get solution dataset part `AdvDrop-main/advdrop_train_wideresnet.npy`
    ```
    cd AdvDrop-main
    python infod_sample.py
    ```
- Change codes at lines 26-29 in 'AdvDrop-main/infod_sample.py'
    ```
    data_npy = '../data_4.npy'
    label_npy = '../label_4.npy'
    arch = 'preactresnet18'
    ```
- Run and get solution dataset part `AdvDrop-main/advdrop_train_preactresnet18.npy`
    ```
    cd AdvDrop-main
    python infod_sample.py
    ```
### 3.3 DeepFool for `data_5.npy` and `data_6.npy`
- Change codes at lines 22-25 in 'DeepFool/Python/test_deepfool.py'
    ```
    data_npy = '../../data_5.npy'
    label_npy = '../../label_5.npy'
    arch = 'wideresnet'
    ```
- Run and get solution dataset part `DeepFool/Python/deepfool_train_wideresnet.npy`
    ```
    cd DeepFool/Python
    python test_deepfool.py
    ```
- Change codes at lines 22-25 in 'DeepFool/Python/test_deepfool.py'
    ```
    data_npy = '../../data_6.npy'
    label_npy = '../../label_6.npy'
    arch = 'preactresnet18'
    ```
- Run and get solution dataset part `DeepFool/Python/deepfool_train_preactresnet18.npy`
    ```
    cd DeepFool/Python
    python test_deepfool.py
    ```
### 3.4 PGD for `data_7.npy` and `data_8.npy`
- Change codes at lines 22-25 in 'pgd/pgd.py'
    ```
    data_npy = '../data_7.npy'
    label_npy = '../label_7.npy'
    arch = 'wideresnet'
    ```
- Run and get solution dataset part `pgd/pgd_train_wideresnet.npy`
    ```
    cd pgd
    python pgd.py
    ```
- Change codes at lines 22-25 in 'pgd/pgd.py'
    ```
    data_npy = '../data_8.npy'
    label_npy = '../label_8.npy'
    arch = 'preactresnet18'
    ```
- Run and get solution dataset part `pgd/pgd_train_preactresnet18.npy`
    ```
    cd pgd
    python pgd.py
    ```

### 3.5 AutoAttack for `data_9.npy` and `data_10.npy`
- Change codes at lines 22-25 in 'aa/aa.py'
    ```
    data_npy = '../data_10.npy'
    label_npy = '../label_10.npy'
    arch = 'wideresnet'
    ```
- Run and get solution dataset part `aa/aa_train_wideresnet.npy`
    ```
    cd aa
    python aa.py
    ```
- Change codes at lines 22-25 in 'aa/aa.py'
    ```
    data_npy = '../data_9.npy'
    label_npy = '../label_9.npy'
    arch = 'preactresnet18'
    ```
- Run and get solution dataset part `aa/aa_train_preactresnet18.npy`
    ```
    cd aa
    python aa.py
    ```
## 4. Combined the Solution Dataset

```
python mix.py
```

The solution dataset will be saved and overwrite the original dataset file `data.npy` and `label.npy`.

Then retrain the two model.

Change code at lines 14 and 29 in `config.py`: `'batch_size': 48`
```
python train.py
```

# Acknowledgments
- Advdrop: https://github.com/RjDuan/AdvDrop
- DeepFool: https://github.com/LTS4/DeepFool
- torchattacks: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/ 
