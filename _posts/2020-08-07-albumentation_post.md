---
title: albumentation 정리
categories:
- Image augmentation
tags:
- Image augmentation
- albumentation
- preprocessing
- Anaconda 3
last_modified_at: 2020-08-018T14:00:00+09:00
toc: true
---
# 1. Intro
이번 대학교 학술제에 발표할 프로젝트를 준비하면서 Image augmentation에 대해 조사하게 되었다.
조사를 진행하면서 Image augmentation에 사용되는 여러 라이브러리와 왜 사용되는지, 무슨 라이브러리가
효율이 좋은지에 대해 알게되어서 이를 포스팅하고자 한다.
# 2. Image augmentation library
많은 Augementation 툴 중에서 albumentation를 사용한 이유는 albumentation이 numpy, Opencv(주된 이유),
imgaug를 기반으로 최적화되어 높은 Performance를 가지고 있기 때문이다. 
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698413-ff7f3700-d95b-11ea-9672-bf1a1812a56c.JPG" alt="Benchmarking"></p>
위의 표는 ImageNet의 validation set에 있는 Image 2000장을 Intel Xeon Platinum 8168 CPU(싱글코어)를 사용해서 transform을 수행한 결과이다.
모든 라이브러리중 albumentation이 굉장히 높은 효율을 보여주고있다. 또한 
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698412-fd1cdd00-d95b-11ea-89dc-a63b3483afe0.JPG" alt="mode1"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698368-c47d0380-d95b-11ea-8e70-19d80cbb92ea.JPG" alt="mode2"></p>
에 대한 표를 살펴보았을때 많은 변형에 대해서 image augmentation 뿐만 아니라 Masks, BBoxes, Keypoints에도 augmentation을 지원하고있다.
# albumentation Content
아래는 프로젝트에서 사용된 augmentation 코드다. 기본적으로 
[bboxes augmentation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)와
[migrating_from_torchvision_to_albumentations](https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/), 
[albumentation 소개 및 사용법](https://hoya012.github.io/blog/albumentation_tutorial/)
등을 참고하여 Anaconda 3 환경에서 코드를 작성했다. bboxes 코드의 경우에는 
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698416-0148fa80-d95c-11ea-9a27-2e5fb319aefb.JPG" alt="bbox_format"></p>
위 이미지에서 보이는 것처럼 각각 다른 서로 다른 bbox format을 가지고 있기 때문에 주의 해줘야한다. 
   
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

#composition set
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
        ], p=1),
        albumentations.OneOf([
            albumentations.RandomBrightness(limit=0.4, p=0.7),
            albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        ], p=0.8),
        albumentations.pytorch.transforms.ToTensor()
    ], bbox_params=albumentations.BboxParams(format='yolo', min_area=4500, min_visibility=0.3))

albumentations_include_rotate = albumentations.Compose([
        albumentations.ToGray(p=0.35),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.OneOf([
            albumentations.HueSaturationValue(p=0.7),
            albumentations.ChannelShuffle(p=0.7),
            albumentations.RGBShift(r_shift_limit =20, g_shift_limit=20,b_shift_limit=20, p=0.4),
        ], p=0.8),
        albumentations.OneOf([
            albumentations.RandomResizedCrop(224, 224, scale=(0.25, 0.25), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(512, 512, scale=(0.25, 0.25), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.RandomResizedCrop(1024, 1024, scale=(0.25, 0.25), ratio=(0.8, 1.2), interpolation=0, p=0.7),
            albumentations.ShiftScaleRotate(border_mode=1, interpolation=3, value=10, p=0.5),
            albumentations.Rotate(border_mode=1,p=0.55),
        ], p=1),
        albumentations.OneOf([
            albumentations.RandomBrightness(limit=0.4, p=0.7),
            albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        ], p=0.8),
        albumentations.OneOf([ 
            albumentations.Resize(224, 224, p=0.7),
            albumentations.Resize(512, 512, p=0.7),
            albumentations.Resize(1024, 1024, p=0.7),
        ], p=1),
        albumentations.pytorch.transforms.ToTensor()
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.4))

def load_bboxes_area(im,bboxes,num_lines): #area, visibility 구함
    area = []
    visib = []
    width, height, channel=im.shape
    resolution=width*height
    for i in range(0,num_lines):
        bbox=list(bboxes[i])
        bbox_width_ratio=bbox[2]
        bbox_height_ratio=bbox[3]
        bbox_width=bbox_width_ratio*width
        bbox_height=bbox_height_ratio*height
        area.append(int(bbox_width)*int(bbox_height))
        visib.append(area[i]/resolution)
    
    return area, visib

def cancel(im,cpath,ccnt): #cancel된 이미지 따로 저장
    f_name= "cancel_data_" + str(ccnt) + ".jpg"
    target = "/canceled_img/"
    path = cpath + target + f_name
    print(path)
    cv2.imwrite(path,im)

def write_train_txt(cpath):
    image_list=[]
    name = "train.txt"
    search = ".jpg"
    
    target = "\\train_set"
    path = cpath + target
    save_path = cpath +"\\"+ name
    
    data_list=os.listdir(path)
    
    for i in data_list:
        if search in i:
            image_list.append(i)
    
    with open(save_path,'w') as f:
        for i in range(0, len(image_list)):
            data = "data/obj/" +str(image_list[i])+"\n"
            f.write(data)
            
def mak_save_img(im,cpath,ite):
    f_name= "train_data_" + str(ite) + ".jpg"
    path = cpath +"/train_set/" + f_name
    cv2.imwrite(path,im)

def mak_plt(im): # plt 그리기
    fig = plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top= 1, hspace= 0, wspace=0)
    plt.imshow(transforms.ToPILImage()(im))
    
def save_plt(ite,cpath): # plt 저장
    path = cpath + "/train_set/"
    f_name = "train_data_" + str(ite) + ".jpg"
    save_path_img = path + f_name
    plt.savefig(save_path_img, bbox_inces='tight',pad_inches=0)
    plt.close()

def save_bboxes_format(ite, transformed_bboxes, cpath, num_lines): # bbox format을 지정한 파일에 저장
    path = cpath + "/train_set/"
    f_name1 = "train_data_" + str(ite) + ".txt"
    save_path_txt = path + f_name1
    
    for i in range(0,num_lines):
        bboxes_txt=list(transformed_bboxes[i])
        temp=bboxes_txt.pop(4)
        bboxes_txt.insert(0,int(temp))
        
        with open(save_path_txt,'a') as f1:
            for s in bboxes_txt:
                f1.write(str(s)+ " ")
            f1.write("\n")
            
def load_bboxes_format(cpath): # bbox format 불러오기
    lines=[]
    yolo_bbox=[]
    yolo_bboxes= []
    with open(cpath,'rt',encoding='UTF8') as f:
        lines=f.readlines()
    for i in range(0,len(lines)):
        for s in lines[i].split(" "):
            yolo_bbox.append(float(s))
        temp=yolo_bbox.pop(0)
        temp=int(temp)
        yolo_bbox.insert(4,str(temp))
        yolo_bboxes.append(yolo_bbox)
        
        yolo_bbox=[]

    return yolo_bboxes, str(temp), len(lines)

def load_info(cnt,cpath): # 정보 불러오기
    image_list = []
    bbox_list = []
    
    target = "\\obj"
    path = cpath + target
    
    data_list=os.listdir(path)
    search = ".jpg"
    search2 = ".JPG"
    for i in data_list:
        if search in i:
            image_list.append(i)
        elif search2 in i:
            image_list.append(i)
        else:
            bbox_list.append(i)
    image_path = path +"\\"+ str(image_list[cnt])
    bbox_path = path +"\\"+ str(bbox_list[cnt])

    bboxes, labels, num_lines = load_bboxes_format(bbox_path)

    return image_path, bboxes, labels, image_list, num_lines

class AlbumentationsDataset(Dataset): # Dataset 지정
    def __init__(self, file_paths, bboxes, labels, transform=None): #Dataset이 다룰 데이터 정의
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
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_t = time.time()
        if self.transform: #augmentation후 augmented된 데이터 반환
            augmented = self.transform(image=image, bboxes=bboxes) 
            image = augmented['image']
            bboxes = augmented['bboxes']
            total_time = (time.time() - start_t)
        return image, bboxes, label, total_time
    
def main():
    area=[]
    visibility=[]
    check=0
    ccnt = 0 # 제외된 이미지 개수
    cnt = 0 # 원본 이미지 개수
    ite= 0 # 복사된 이미지 개수
    cpath=os.getcwd()
    image_path, bbox, label, images, num_lines=load_info(cnt,cpath)
    

    for i in range(0, len(images)):
    #for i in range(0,20):
        # 이미지, 경계박스 정보 불러오기
        image_path, bbox, label, ims, num_lines =load_info(cnt,cpath)

        cnt+=1 #obj 폴더 이미지 카운터    
        iterations = 25 # 한 그림에 대한 반복횟수
        for i in range(0, iterations):
            #dataset 불러오기
            albumentations_dataset=AlbumentationsDataset(
                file_paths=image_path,
                bboxes=bbox,
                labels=[label],
                transform=albumentations_include_rotate,
            )
            transformed_image, transformed_bboxes, label, transform_time = albumentations_dataset[ite]
            im=transformed_image.numpy() # tensor -> numpy
            im=(im * 255).round().astype(np.uint8) # float32 -> uint8
            im=np.transpose(im,(1,2,0)) # shape 변형
            
            # plt 여백 및 축 제거, img 저장 % rotate포함한 composition set을 가지고 변형하면 안됨
            #mak_plt(transformed_image)# RGB COLOR
            #save_plt(ite,cpath)
            
            if len(transformed_bboxes) != 0:
                area, visibility=load_bboxes_area(im,transformed_bboxes,num_lines)
            
            # 이미지, bbox 저장
            if int(len(transformed_bboxes)) == int(num_lines):
                for i in range(0,num_lines):
                    if visibility[i] > 0.5:
                        check+=1
                if check == num_lines:
                mak_save_img(im,cpath,ite)# BGR COLOR / cv2.imwrite 저장
                save_bboxes_format(ite, transformed_bboxes,cpath,num_lines)
                ite+=1 # 복제된 모든 이미지 카운터
                check=0
            else:
                #cancel(im,cpath,ccnt)
                i-=1
                ccnt+=1
            area=[]
            visibility=[]
            
    write_train_txt(cpath) # train.txt 작성
    
main()
```
# 3. 문제 및 해결
### 3.1. plt 이미지 저장시 여백 문제
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90479766-26164c80-e16a-11ea-91bc-2b5b0043e681.PNG" alt="plt_savefig_image"></p>
rotate를 포함해서 augmentation을 진행하고 plt.savefig함수로 이미지를 저장할시 이미지의 위 아래가 흰색 여백으로 채워지는 오류가 있다. 이 오류는  savefig함수를 사용하는 대신 cv2.imwrite함수를 사용해 해결했다.
또한 plt.savefig을 사용할 경우 저장한 이미지의 figure를 초기화 시켜주지 않으면 augmentation 진행하다가 Out of memory날 수 있다.

  
### 3.2. scale 범위를 넘는 값 수정
소스코드중 bbox를 normalize(albumentation -> yolo)하는 과정에서 
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698417-0312be00-d95c-11ea-9f08-61e82d2977a5.PNG" alt="mode_error"></p>
위와 같은 오류를 만났는데 소수점 연산하다가 우연찮게 발생된거라 판단해서 albumentations/augmentations/bbox_utils.py에 있는 normalize(bbox, rows, cols) 함수를 약간 수정했다.  


### 3.3. 흑색 이미지
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90479310-6f19d100-e169-11ea-8c18-a68511d54ebe.png" alt="black_image"></p>
plt.savefig함수로 이미지 저장하는 문제를 해결하기 위해 cv2.imwrite함수를 사용하였다. 주의할 것은 cv2.imwrite는 행렬 요소의 데이터 타입이 uint8을 지원하기때문에 만약 dtype이 float32나 float64같은 소수점 타입이면 tensorflow나 numpy라이브러리에서
dtype변형을 지원해주는 함수를 사용해 해결해야한다. 저자는 numpy라이브러리에서 지원하는 아래의 함수를 사용해 해결하였다.

  
```python
im=(im * 255).round().astype(np.uint8)
```  


### 3.4. 학습하기에 적합하지 않은 이미지 분류
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90480075-b0f74700-e16a-11ea-95f6-8e38c591a408.png" alt="area and vis"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90480083-b359a100-e16a-11ea-8f14-5eadc354bb8a.png" alt="after"></p>
albumentation라이브러리에서는 학습을 진행하기에 적합하지 않은 이미지를 걸러내기 위하여 min_visibility와 min_area기능을 지원한다고 나와있다. 근데 정상 작동하지 않아서 
아래 코드를 사용해 bbox area와 visibility(albumentation이 지원하는 visibility하고는 다름)를 구하고 main에서 조건문으로 기준 이하면 해당 이미지를 cancel시켰다.

```python
def load_bboxes_area(im,bboxes,num_lines):
    area = []
    visib = []
    width, height, channel=im.shape
    resolution=width*height
    for i in range(0,num_lines):
        bbox=list(bboxes[i])
        bbox_width_ratio=bbox[2]
        bbox_height_ratio=bbox[3]
        bbox_width=bbox_width_ratio*width
        bbox_height=bbox_height_ratio*height
        area.append(int(bbox_width)*int(bbox_height))
        visib.append(area[i]/resolution)
    
    return area, visib
```

# 4. Data Preprocessing & Augmentation
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90480859-d769b200-e16b-11ea-9b7b-7cc7a6f0b3cd.JPG" alt="conv single"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90481003-097b1400-e16c-11ea-9b20-b54d36228fc3.JPG" alt="conv multi"></p>
VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION논문에서는 ILSVRC(training:1.3M images-1000classes, val:50K images, testing:100K images-held out label)으로 학습을 진행 시켰을때 test set과 
train set 모두 다른 스케일을 줄시에 분류 오류가 줄어드는 경향을 보여준다고 나와있다.
만약 albumentation에서 해당 효과를 보고 싶으면 Resize 메소드를 composition set에 포함시키면된다.
# 5. Outro
IPL에서 처음 수행하게된 일이였는데 배울게 많았다. 나중에 imgaug라이브러리를 사용해서 라이브러리간의 augmentation시켰을시 얼마나 차이나는지 살펴보고싶다.