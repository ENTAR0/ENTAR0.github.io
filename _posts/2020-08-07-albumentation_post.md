---
title: albumentation 정리
categories:
- Image augmentation
tags:
- Image augmentation
- albumentation
- preprocessing
- Anaconda 3
last_modified_at: 2020-08-20T14:00:00+09:00
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


# 3. albumentation Code
[bboxes augmentation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)와[migrating_from_torchvision_to_albumentations](https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/), 
[albumentation 소개 및 사용법](https://hoya012.github.io/blog/albumentation_tutorial/)등을 참고하여 Anaconda 3 환경에서 코드를 작성했다. 
만약 코드가 보고싶다면 [augmentation_tool](https://github.com/ENTAR0/ENATR0/blob/master/renewing%20augmentation.py)를 참고하면 된다. 
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90720198-18d79a00-e2f1-11ea-9f7f-3e8e58bca70e.png" alt="architecture"></p> 
이 이미지는 코드의 구조를 간략하게 소개한 것이다.
 albumentation 라이브러리를 사용해서 bbox를 포함한  augmentation을 진행할때
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698416-0148fa80-d95c-11ea-9a27-2e5fb319aefb.JPG" alt="bbox_format"></p>
위 이미지에서 보이는 것처럼 bbox들은 서로 다른 bbox format을 가지고 있기 때문에 주의 해줘야한다. 


# 4. 문제 및 해결
### 4.1. plt 이미지 저장시 여백 문제
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90479766-26164c80-e16a-11ea-91bc-2b5b0043e681.PNG" alt="plt_savefig_image"></p>
rotate를 포함해서 augmentation을 진행하고 plt.savefig함수로 이미지를 저장할시 이미지의 위 아래가 흰색 여백으로 채워지는 오류가 있다. 이 오류는  savefig함수를 사용하는 대신 cv2.imwrite함수를 사용해 해결했다.
또한 plt.savefig을 사용할 경우 저장한 이미지의 figure를 초기화 시켜주지 않으면 augmentation 진행하다가 Out of memory날 수 있다.

  
### 4.2. scale 범위를 넘는 값 수정
소스코드중 bbox를 normalize(albumentation -> yolo)하는 과정에서 
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/89698417-0312be00-d95c-11ea-9f08-61e82d2977a5.PNG" alt="mode_error"></p>
위와 같은 오류를 만났는데 소수점 연산하다가 우연찮게 발생된거라 판단해서 albumentations/augmentations/bbox_utils.py에 있는 normalize(bbox, rows, cols) 함수를 약간 수정했다.  


### 4.3. 흑색 이미지
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90479310-6f19d100-e169-11ea-8c18-a68511d54ebe.png" alt="black_image"></p>
plt.savefig함수로 이미지 저장하는 문제를 해결하기 위해 cv2.imwrite함수를 사용하였다. 주의할 것은 cv2.imwrite는 행렬 요소의 데이터 타입이 uint8을 지원하기때문에 만약 dtype이 float32나 float64같은 소수점 타입이면 tensorflow나 numpy라이브러리에서
dtype변형을 지원해주는 함수를 사용해 해결해야한다. 저자는 numpy라이브러리에서 지원하는 아래의 함수를 사용해 해결하였다.

  
```python
im=(im * 255).round().astype(np.uint8)
```  


### 4.4. 학습하기에 적합하지 않은 이미지 분류
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


# 5. Data Preprocessing & Augmentation
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90480859-d769b200-e16b-11ea-9b7b-7cc7a6f0b3cd.JPG" alt="conv single"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90481003-097b1400-e16c-11ea-9b20-b54d36228fc3.JPG" alt="conv multi"></p>
VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION논문에서는 ILSVRC(training:1.3M images-1000classes, val:50K images, testing:100K images-held out label)으로 학습을 진행 시켰을때 test set과 
train set 모두 다른 스케일을 줄시에 분류 오류가 줄어드는 경향을 보여준다고 나와있다.
만약 albumentation에서 해당 효과를 보고 싶으면 Resize 메소드를 composition set에 포함시키면된다.


# 6. Outro
IPL에서 처음 수행하게된 일이였는데 배울게 많았다. 나중에 imgaug라이브러리를 사용해서 라이브러리간의 augmentation시켰을시 얼마나 차이나는지 살펴보고싶다.