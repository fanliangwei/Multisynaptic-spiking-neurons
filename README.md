# MSF neurons Implementation

This code implements the MSF neurons for various tasks. We select some typical training codes for tasks in the paper to present.


## 📁 Datasets Preparation
| Dataset      | Source                                                                                     |
|--------------|-------------------------------------------------------------------------------------------|
| CIFAR         | [CIFAR Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)                           |
| ImageNet-1k         | [ImageNet Dataset](https://www.image-net.org/)                           |
| COCO2017         | [COCO Dataset](https://cocodataset.org/#home)                           |
| Gen1         | [Gen1 Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)    
| SHD    | [Spiking Heidelberg Datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) |
| DEAP         | [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)                           |
| MMIDB    | [MMIDB Dataset](https://physionet.org/content/eegmmidb/1.0.0/) |
| Sleep-EDF-18         | [Sleep-EDF-18 Dataset](https://www.physionet.org/content/sleep-edfx/1.0.0/)                           |
## ⚙️ Prerequisites
Python >=3.8
torch==1.11.0
cupy==12.3.0
spikingjelly==0.0.0.0.12
timm==0.9.16
## 🚀 Quick Start
### 1. Data Preprocessing

Classification:

ImageNet with the following folder structure
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

Detection:

### 2. Model Training and Testing
#### Example for the recognition task on CIFAR-10:
Training:
```
cd CIFAR-10
python train.py
```
Testing:
Download the trained model first [MHSANet-29](https://drive.google.com/file/d/1qdX1wh_1ywvYRDFsgbGkKrzkkHuKBZXi/view?usp=drive_link)
```
cd CIFAR-10
python test.py
```
#### Example for the recognition task on ImageNet-1k:
Training:
```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```



## Acknowledgement
Related project:



