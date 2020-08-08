---
title: albumentation 정리
categories:
- Image augmentation
tags:
- Image augmentation
- preprocessing
- Anaconda 3
last_modified_at: 2020-08-08T14:00:00+09:00
toc: true
---
# Intro
이번 대학교 학술제에 발표할 프로젝트를 준비하면서 Image augmentation에 대해 조사하게 되었다.
조사를 진행하면서 Image augmentation에 사용되는 여러 라이브러리와 왜 사용되는지, 무슨 라이브러리가
효율이 좋은지에 대해 알게되어서 이를 포스팅하고자 한다.
# Image augmentation library
많은 Augementation 툴 중에서 albumentation를 사용한 이유는 albumentation이 numpy, Opencv(주된 이유),
imgaug를 기반으로 최적화되어 높은 Performance를 가지고 있기 때문이다. ![Benchmarking](./assets/images/Benchmarking.jpg){: .align -center}
위의 표는 ImageNet의 validation set에 있는 Image 2000장을 Intel Xeon Platinum 8168 CPU(싱글코어)를 사용해서 transform을 수행한 결과이다.
모든 라이브러리중 albumentation이 굉장히 높은 효율을 보여주고있다. 또한 ![mode](/assets/images/mode1.jpg){: .align -center}
![mode](/assets/images/mode2.jpg){: .align -center}에 대한 표를 살펴보았을때
많은 변형에 대해서 image augmentation 뿐만 아니라 Masks, BBoxes, Keypoints에도 augmentation을 지원하고있다.
# albumentation Content
아래는 프로젝트에서 사용된 augmentation 코드다. 기본적으로 [bboxes augmentation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)와
[migrating_from_torchvision_to_albumentations](https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/), [albumentation 소개 및 사용법](https://hoya012.github.io/blog/albumentation_tutorial/)등을 참고하여
Anaconda 3 환경에서 코드를 작성했다. bboxes 코드의 경우에는 ![bbox format](/assets/images/bbox_format.jpg){: .align -center}위 이미지에서 보이는 것처럼 각각 다른 서로 다른
bbox format을 가지고 있기 때문에 주의 해줘야한다. 또한 본인은 소스코드중 bbox를 denormalize(albumentation -> yolo)하는 과정에서 ![error](/assets/images/bbox_error.png){: .align -center}
위와 같은 오류를 만났는데 소수점 연산하다가 우연찮게 발생된거라 판단해서 albumentations/augmentations/bbox_utils.py에 있는 denormalize(bbox, rows, cols) 함수를 약간 수정했다.   
```python
from PIL import Image
import os
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

albumentations_transform = albumentations.Compose([
        albumentations.ToGray(p=0.35),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.OneOf([
            albumentations.HueSaturationValue(p=0.7),
            albumentations.ChannelShuffle(p=0.7),
            albumentations.RGBShift(r_shift_limit =20, g_shift_limit=20,b_shift_limit=20, p=0.4),
        ], p=0.8),
        albumentations.OneOf([
            albumentations.RandomResizedCrop(224, 224, scale=(0.25, 0.9), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(512, 512, scale=(0.25, 0.9), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(1024, 1024, scale=(0.25, 0.9), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.ShiftScaleRotate(p=0.4),
        ], p=1),
        albumentations.OneOf([
            albumentations.RandomBrightness(limit=0.4, p=0.7),
            albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        ], p=0.8),
        albumentations.pytorch.transforms.ToTensor()
    ], bbox_params=albumentations.BboxParams(format='yolo', min_area=0, min_visibility=0))

def write_train_txt():
    image_list=[]
    path = os.getcwd()
    name = "train.txt"
    search = ".jpg"
    
    address = os.getcwd()
    target = "\\train_set"
    path = address + target
    save_path = address +"\\"+ name
    
    data_list=os.listdir(path)
    
    for i in data_list:
        if search in i:
            image_list.append(i)
    
    with open(save_path,'w') as f:
        for i in range(0, len(image_list)):
            data = "data/obj/" +str(image_list[i])+"\n"
            f.write(data)

def mak_plt(im): # plt 그리기
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top= 1, hspace= 0, wspace=0)
    plt.imshow(im)
    
def save_plt(): # plt 저장
    path = os.getcwd()
    path = path + "/train_set/"
    f_name2 = "train_data_" + str(ite) + ".jpg"
    save_path_img = path + f_name2
    plt.savefig(save_path_img, bbox_inces='tight',pad_inches=0)
    plt.close()

def save_bbox_format(): # bbox format을 지정한 파일에 저장
    path = os.getcwd()
    path = path + "/train_set/"
    f_name1 = "train_data_" + str(ite) + ".txt"
    save_path_txt = path + f_name1
    
    bboxes_txt=list(transformed_bboxes[0])
    temp=bboxes_txt.pop(4)
    bboxes_txt.insert(0,int(temp))
        
    with open(save_path_txt,'w') as f1:
        for s in bboxes_txt:
            f1.write(str(s)+ " ")
            
def load_bbox_format(t_path): # bbox format 불러오기
    yolo_bboxes= []
    with open(t_path,'r') as f:
        line=f.readline()
        for s in line.split(" "):
            yolo_bboxes.append(float(s))
    temp=yolo_bboxes.pop(0)
    temp=int(temp)
    yolo_bboxes.insert(4,str(temp))

    return yolo_bboxes, str(temp)

def load_info(seq): # 정보 불러오기
    image_list = []
    bbox_list = []
    
    address = os.getcwd()
    target = "\\obj"
    path = address + target
    
    data_list=os.listdir(path)
    
    search = ".jpg"
    for i in data_list:
        if search in i:
            image_list.append(i)
        else:
            bbox_list.append(i)

    image_path = path +"\\"+ str(image_list[seq])
    bbox_path = path +"\\"+ str(bbox_list[seq])

    bboxes, labels = load_bbox_format(bbox_path)

    return image_path, bboxes, labels, image_list

class AlbumentationsDataset(Dataset): # Dataset 지정
    def __init__(self, file_paths, bboxes, labels, transform=None):
        print(bboxes)
        self.file_paths = file_paths
        self.labels = labels
        self.bboxes = bboxes
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels
        file_path = self.file_paths
        bboxes = self.bboxes
        image = cv2.imread(file_path)
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_t = time.time()
        if self.transform:
            augmented = self.transform(image=image, bboxes=[bboxes]) 
            image = augmented['image']
            bboxes = augmented['bboxes']
            total_time = (time.time() - start_t)
        return image, bboxes, label, total_time
    
def main():
    cnt = 0
    ite= 0
    image_path, bbox, label, images=load_info(cnt)

    for i in range(0, len(images)):
    #for i in range(0,20):
        # 이미지, 경계박스 정보 불러오기
        image_path, bbox, label, ims =load_info(cnt)

        cnt+=1 #obj 폴더 이미지 카운터    
        # 한 그림에 대한 반복횟수
        iterations = 1
        for i in range(0, iterations):
            #dataset 불러오기
            albumentations_dataset=AlbumentationsDataset(
                file_paths=image_path,
                bboxes=bbox,
                labels=[label],
                transform=albumentations_transform,
            )
            transformed_image, transformed_bboxes, label, transform_time = albumentations_dataset[ite]
        
            im=transformed_image.numpy() # tensor -> numpy
            im=np.transpose(im,(1,2,0)) # shape 변형
        
            # plt 여백 및 축 제거
            mak_plt(im)
        
            # 이미지 및 텍스트 파일 저장
            save_bbox_format()
            save_plt()

            ite+=1 # 복제된 모든 이미지 카운터
    write_train_txt() # train.txt 작성
    
main()
```
# Data Preprocessing & Augmentation
추후 포스팅 예정
# Outro
추후 포스팅