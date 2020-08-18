---
title: albuementation method 정리
categories:
- Image augementation
last_modified_at: 2020-08-018T14:00:00+09:00
tags:
- Image augmentation
- albumentation
toc: true
---
# Intro
albumentation을 사용해서 augmentation을 진행하는데 composition set에 들어가는 옵션의 매개변수를 수정할시
얼마나 어떻게 변하는지 시각적으로 보여주고 어떤 값들을 가지는지 명시하고자 작성했습니다.
albumentation 공식 Documentation인 [albumentations.augmentations](https://albumentations.readthedocs.io/en/latest/api/augmentations.html)와
[albumentations.augmentations.transforms source code](https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html)
를 참고하여 작성했습니다.


# albumentation (작성중)
```python
transforms.Blur(blur_limit(int, (int, int)), always_apply=(bool), p=(float))
```
```python
transforms.VerticalFlip(always_apply=(bool), p=(float))
```
```python
transforms.HorizontalFlip(always_apply=(bool), p=(float))
```
```python
transforms.Flip(always_apply=(bool), p=(float))
```
```python
transforms.Normalize(mean=(float, list of float), std=(float, list of float), max_pixel_value=(float), 
always_apply=(bool), p=(float))
```
```python
transforms.Transpose(always_apply=(bool), p=(float))
```
```python
transforms.RandomCrop(height=(int), width=(int), always_apply=(bool), p=(float))
```
```python
transforms.RandomGamma(gamma_limit=(float or (float, float)), eps=None, always_apply=(bool), p=(float))
```
```python
transforms.RandomRotate90(always_apply=(bool), p=(float))
```
```python
transforms.Rotate(limit=((int, int) or int), interpolation=(OpenCV flag), border_mode=(OpenCV flag), 
value=(int, float, list of ints, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```
```python
transforms.ShiftScaleRotate(shift_limit=((float, float) or float), scale_limit((float, float) or float), rotate_limit((int, int or int), 
interpolation=(OpenCV flag), border_mode(OpenCV flag), value(int, float, list of int, list of float), mask_value=(int, float), 
always_apply=(bool), p=(float))
```
```python
transforms.CenterCrop(height=(int), width=(int), always_apply=(bool), p=(float))
```
```python
transforms.OpticalDistortion(distort_limit=(float, (float, float)), shift_limit=(float, (float, float)), 
interpolation=(OpenCV flag), border_mode=(OpenCV flag), value=(int, float, list of ints, list of float), 
mask_value=(int, float), always_apply=(bool), p=(float))
```
```python
transforms.GridDistortion(num_steps=(int), distort_limit=(float, (float, float)), interpolation=(OpenCV flag), border_mode=(OpenCV flag), 
value=(int, float, list of ints, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```
```python
transforms.ElasticTransform(alpha=(float), sigma=(float), alpha_affine=(float), interpolation=(OpenCV flag), border_mode=(OpenCV flag), 
value=(int, float, list of ints, list of float), mask_value=(int, float), approximate=(bool), always_apply=(bool), p=(float))
```
```python
transforms.RandomGridShuffle(grid=((int, int) always_apply=(bool), p=(float))
```
```python
transforms.HueSaturationValue(hue_shift_limit=((int, int) or int), sat_shift_limit=((int, int) or int), val_shift_limit=((int, int) or int), always_apply=(bool), p=(float))
```
```python
transforms.PadIfNeeded(min_height=(int), min_width=(int), border_mode=(OpenCV flag), value=(int, float, list of int, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```
```python
transforms.RGBshift(r_shift_limit=((int, int) or int), g_shift_limit=((int, int) or int), b_shift_limit=((int, int) or int), p=(float))
```
```python
transforms.RandomBrightness(limit=((float, float) or float), always_apply=(bool), p=(float))
```
```python
transforms.RandomContrast(limit=((float, float) or float), always_apply=(bool), p=(float))
```
```python
transforms.MotionBlur(blur_limit=(int), always_apply=(bool), p=(float))
```
```python
transforms.MedianBlur(blur_limit=(int), always_apply=(bool), p=(float))
```
```python
tranforms.GaussianBlur(blur_limit=(int), always_apply=(bool), p=(float))
```
```python
transforms.GaussNoise(var_limit=((float, float) or float), mean=(float), always_apply=(bool), p=(float))
```
```python
transforms.GlassBlur(sigma=(float), max_delta=(int), iterations=(int), mode=(str), always_apply=(bool), p=(float))
```
```python
transforms.CLAHE(clip_limit=(float or (float, float)), tile_grid_size=((int, int)), always_apply=(bool), p=(float))
```