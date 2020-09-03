---
title: albuementation method 정리
categories:
- Image augmentation
last_modified_at: 2020-08-29T14:00:00+09:00
tags:
- Image augmentation
- albumentation
toc: true
---
# 1. Intro
albumentation을 사용해서 augmentation을 진행하는데 composition set에 들어가는 옵션의 매개변수를수정할시
얼마나 어떻게 변하는지 시각적으로 보여주고 어떤 값들을 가지는지 명시하고자 작성했습니다.
albumentation 공식 Documentation인 [albumentations.augmentations](https://albumentations.readthedocs.io/en/latest/api/augmentations.html)와
[albumentations.augmentations.transforms source code](https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html)
를 참고하여 작성했습니다.


# 2. 자주 사용되는 매개변수들
- always_apply=(Boolean) : Default값은 False로 설정되있는데 딱히 신경안써도 됩니다.
- p=(float) : 해당 augmentation option을 몇퍼센트의 확률로 적용할지 결정합니다. (ex. p=0.5면 50%확률)
- interpolation=(OpenCV flag) : cv2라이브러리에서 제공하는 보간법(interpolation)을 적용합니다.
  - OpenCV flag=0 : cv2.INTER_NEAREST 적용
  - OpenCV flag=1 : cv2.INTER_LINEAR 적용
  - OpenCV flag=2 : cv2.INTER_CUBIC 적용
  - OpenCV flag=3 : cv2.INTER_AREA 적용
  - OpenCV flag=4 : cv2.INTER_LANCZOS4 적용
- border_mode=(OpenCV flag) : cv2라이브러리에서 제공하는 외간법(extrapolation)을 적용합니다.
  - OpenCV flag=0 : cv2.BORDER_CONSTANT 적용
  - OpenCV flag=1 : cv2.BORDER_REPLICATE 적용 - 추천
  - OpenCV flag=2 : cv2.BORDER_REFLECT 적용
  - OpenCV flag=3 : cv2.BORDER_WRAP 적용
  - OpenCV flag=4 : cv2.BORDER_REFLECT_101 적용


# 3. albumentation methods(작성중)
#### 1. Blur
임의의 크기의 커널을 이용해 이미지를 흐릿하게 만듭니다.
```python
class albumentations.augmentations.transforms.Blur(blur_limit(int, (int, int)), always_apply=(bool), p=(float))
```


#### 2. VerticalFlip
이미지를 수직으로 뒤집습니다.
```python
class albumentations.augmentations.transforms.VerticalFlip(always_apply=(bool), p=(float))
```


#### 3. HorizontalFlip
이미지를 수평으로 뒤집습니다.
```python
class albumentations.augmentations.transforms.HorizontalFlip(always_apply=(bool), p=(float))
```


#### 4. Flip
수직 뒤집기 또는 수평 뒤집기 또는 둘 다 적용합니다.
```python
class albumentations.augmentations.transforms.Flip(always_apply=(bool), p=(float))
```


#### 5. Normalize
픽셀 값을 255로 나눕니다.
```python
class albumentations.augmentations.transforms.Normalize(mean=(float, list of float), std=(float, list of float), max_pixel_value=(float),
always_apply=(bool), p=(float))
```


#### 6. Transpose
이미지의 행과열을 바꿔놓습니다.
```python
class albumentations.augmentations.transforms.Transpose(always_apply=(bool), p=(float))
```


#### 7. RandomCrop
이미지의 임의의 부분에 대해 Crop을 수행합니다.
```python
class albumentations.augmentations.transforms.RandomCrop(height=(int), width=(int), always_apply=(bool), p=(float))
```


#### 8. RandomGamma
임의의 감마처리를 수행합니다.
```python
class albumentations.augmentations.transforms.RandomGamma(gamma_limit=(float or (float, float)), eps=None, always_apply=(bool), p=(float))
```


#### 9. RandomRotate90
무작위로 90도 회전시킵니다. 0도 돌아갈 수 도 있고 270도 돌아갈 수 도 있습니다.
```python
class albumentations.augmentations.transforms.RandomRotate90(always_apply=(bool), p=(float))
```


#### 10. Rotate
limit 매개변수 분포에 적절한 각도를 회전합니다.
```python
class albumentations.augmentations.transforms.Rotate(limit=((int, int) or int), interpolation=(OpenCV flag), border_mode=(OpenCV flag),
value=(int, float, list of ints, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 11. ShiftScaleRotate
무작위로 아핀 변환을 적용합니다.
```python
class albumentations.augmentations.transforms.ShiftScaleRotate(shift_limit=((float, float) or float), scale_limit((float, float) or float), rotate_limit((int, int or int),
interpolation=(OpenCV flag), border_mode(OpenCV flag), value(int, float, list of int, list of float), mask_value=(int, float),
always_apply=(bool), p=(float))
```


#### 12. CenterCrop
이미지의 중앙부분에 대해 Crop을 수행합니다. (만약 사용한다면 uint8 dtype을 가진 이미지에 수행하는걸 추천합니다.)
```python
class albumentations.augmentations.transforms.CenterCrop(height=(int), width=(int), always_apply=(bool), p=(float))
```


#### 13. OpticalDistortion
이미지에 대해 시각적인 왜곡을 줍니다.
```python
class albumentations.augmentations.transforms.OpticalDistortion(distort_limit=(float, (float, float)), shift_limit=(float, (float, float)),
interpolation=(OpenCV flag), border_mode=(OpenCV flag), value=(int, float, list of ints, list of float),
mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 14. GridDistortion
이미지를 Grid로 나눠서 왜곡합니다.
```python
class albumentations.augmentations.transforms.GridDistortion(num_steps=(int), distort_limit=(float, (float, float)), interpolation=(OpenCV flag), border_mode=(OpenCV flag),
value=(int, float, list of ints, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 15. ElasticTransform
이미지를 Grid로 나눠서 Grid마다 탄성있는 왜곡을 줍니다.
```python
class albumentations.augmentations.transforms.ElasticTransform(alpha=(float), sigma=(float), alpha_affine=(float), interpolation=(OpenCV flag), border_mode=(OpenCV flag),
value=(int, float, list of ints, list of float), mask_value=(int, float), approximate=(bool), always_apply=(bool), p=(float))
```


#### 16. RandomGridShuffle
이미지를 Grid로 나눠서 임의의 Grid cell을 섞습니다.
```python
class albumentations.augmentations.transforms.RandomGridShuffle(grid=((int, int) always_apply=(bool), p=(float))
```


#### 17. HueSaturationValue
이미지의 색조와 채도의 값을 임의의 값으로 변경합니다.
```python
class albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=((int, int) or int), sat_shift_limit=((int, int) or int), val_shift_limit=((int, int) or int), always_apply=(bool), p=(float))
```


#### 18. PadIfNeeded
이미지의 가장자리에 Pad를 추가합니다.
```python
class albumentations.augmentations.transforms.PadIfNeeded(min_height=(int), min_width=(int), border_mode=(OpenCV flag), value=(int, float, list of int, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 19. RGBshift
RGB채널의 값을 임의의 값으로 변경합니다.
```python
class albumentations.augmentations.transforms.RGBshift(r_shift_limit=((int, int) or int), g_shift_limit=((int, int) or int), b_shift_limit=((int, int) or int), p=(float))
```


#### 20. RandomBrightness
이미지의 밝기는 임의의 값으로 변경합니다.
```python
class albumentations.augmentations.transforms.RandomBrightness(limit=((float, float) or float), always_apply=(bool), p=(float))
```


#### 21. RandomContrast
이미지의 Contrast를 임의의 값으로 변경합니다. (범위가 양수면 어두운 부분은 더 어둡게 밝은 부분은 더 밝게 범위가 음수면 그 반대)
```python
class albumentations.augmentations.transforms.RandomContrast(limit=((float, float) or float), always_apply=(bool), p=(float))
```


#### 22. MotionBlur
임의의 크기의 커널을 이용해 이미지에 모션블러를 적용합니다.
```python
class albumentations.augmentations.transforms.MotionBlur(blur_limit=(int), always_apply=(bool), p=(float))
```


#### 23. MedianBlur
미디언 필터를 적용해 흐릿하게 만듭니다.
```python
class albumentations.augmentations.transforms.MedianBlur(blur_limit=(int), always_apply=(bool), p=(float))
```


#### 24. GaussianBlur
가우시안 필터를 적용해 흐릿하게 만듭니다.
```python
class albumentations.augmentations.tranforms.GaussianBlur(blur_limit=(int), always_apply=(bool), p=(float))
```


#### 25. GaussNoise
가우시안 노이즈를 이미지에 적용합니다.
```python
class albumentations.augmentations.transforms.GaussNoise(var_limit=((float, float) or float), mean=(float), always_apply=(bool), p=(float))
```


#### 26. GlassBlur
글래스 노이즈를 이미지에 적용합니다.
```python
class albumentations.augmentations.transforms.GlassBlur(sigma=(float), max_delta=(int), iterations=(int), mode=(str), always_apply=(bool), p=(float))
```


#### 27. CLAHE
이미지의 Contrast를 적절하게 변화시켜줍니다.
```python
class albumentations.augmentations.transforms.CLAHE(clip_limit=(float or (float, float)), tile_grid_size=((int, int)), always_apply=(bool), p=(float))
```


#### 28.ChannelShuffle
RGB이미지의 채널을 재정렬 합니다.
```python
class albumentations.augmentations.transforms.ChannelShuffle(always_apply=(bool), p=(float))
```


#### 29. InverImg
픽셀 값에서 255를 빼서 색상을 반전시킵니다.
```python
class albumentations.augmentations.transforms.InvertImg(always_apply=(bool), p=(float))
```


#### 30. ToGray
이미지를 그레이스케일로 변환합니다.
```python
class albumentations.augmentations.transforms.ToGray(always_apply=(bool), p=(float))
```


#### 31. ToSepia
세피아 필터를 이미지에 적용합니다.
```python
class albumentations.augmentations.transforms.ToSepia(always_apply=(bool), p=(float))
```


#### 32. JpegCompression
Jpeg압축률을 낮춥니다.
```python
class albumentations.augmentations.transforms.JpegCompression(quality_lower=(float), quality_upper=(float), always_apply=(bool), p=(float))
```


#### 33. ImageCompression
Jpeg, WebP압축률을 낮춥니다.
```python
class albumentations.augmentations.transforms.ImageCompression(quality_lower=(float), quality_upper=(float), compression_type=(ImageCompressionType), always_apply=(bool), p=(float))
```


#### 34. Cutout
이미지에서 num_holes 만큼의 정사각형 구역을 제거합니다.
```python
class albumentations.augmentations.transforms.Cutout(num_holes=(int), max_h_size=(int), max_w_size=(int), fill_value=(int, float, list of int, list of float), always_apply=(bool), p=(float))
```


#### 35. CoarseDropout
이미지에서 min~max만큼의 직사각형 구역을 제거합니다.
```python
class albumentations.augmentations.transforms.CoarseDropout(max_holes=(int), max_height=(int), max_width=(int), min_holes=(int), min_height=(int), min_width=(int), fill_value=(int, float, list of int, list of float), always_apply=(bool), p=(float))
```


#### 36. ToFloat
픽셀 값을 float 32 dtype으로 변환합니다.
```python
class albumentations.augmentations.transforms.ToFloat(max_value=(float), always_apply=(bool), p=(float))
```


#### 37. FromFloat
float 32 dtype을 다른 dtype으로 변환합니다. 기본적으로 uint 16으로 변환합니다.
```python
class albumentations.augmentations.transforms.FromFloat(max_value=(float),  dtype=(string or numpy data type), always_apply=(bool), p=(float))
```


#### 38. Crop
구역을 정해서 그 구역에 대해 Crop을 수행합니다.
```python
class albumentations.augmentations.transforms.Crop(x_min=(int), y_min=(int), x_max=(int), y_max=(int), always_apply=(bool), p=(float))
```


#### 39. CropNonEmptyMaskIfExists
mask를 가진 구역에 대해 Crop을 수행합니다. 만약 그 구역에 mask가 존재 하지않는다면 임의의 구역에 대해 Crop을 수행합니다.
```python
class albumentations.augmentations.transforms.CropNonEmptyMaskIfExists(height=(int), width=(int), ignore_values=(list of int), ignore_channels=(list of int), always_apply=(bool), p=(float))\
```


#### 40. RandomScale
임의의 크기로 이미지를 Rescale합니다.
```python
class albumentations.augmentations.transforms.RandomScale(scale_limit=((float, float), or float), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```

#### 41. SamllestaMaxSize
처음 이미지의 비율을 유지한체 minimum side를 max_size와 같게 Rescale를 수행합니다.
```python
class albumentations.augmentations.transforms.SmallestMaxSize(max_size=(int), interpolation=(OpneCV flag), always_apply=(bool), p=(float))
```


#### 42. Resize
폭과 높이를 주어진 값으로 재조정합니다.
```python
class albumentations.augmentations.transforms.Resize(height=(int), width=(int), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```


#### 43. RandomResizedCrop
임의의 부분을 Crop하고 주어진 값으로 rescale합니다.
```python
class albumentations.augmentations.transforms.RandomResizedCrop(height=(int), width=(int), scale=((float, float)), ratio=((float, float)), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```


#### 44. RandomBrightnessContrast
Brightness와 Contrast를 임의의 값으로 변경합니다.
```python
class albumentations.augmentations.transforms.RandomBrightnessContrast(brightness_limit=((float, float) or float), contrast_limit=((float, float), or float), brightness_by_max=(Boolean), always_apply=(bool), p=(float))
```


#### 45. RandomCropNearBBox
bbox 위치를 탐색하고 탐색한 위치의 x,y좌표를 임의로 변경해서 그 부분을 Crop 합니다.
```python
class albumentations.augmentations.transforms.RandomCropNearBBox(max_part_shift=(float), always_apply=(bool), p=(float))
```


#### 46. RandomSizedBBoxSafeCrop
bbox 손실 없이 임의의 부분을 Crop 하고 rescale합니다.
```python
class albumentations.augmentations.transforms.RandomSizedBBoxSafeCrop(height=(int), width=(int), erosion_rate=(float), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```


### 47. RandomSnow

```python
class albumentations.augmentations.transforms.RandomSnow(snow_point_lower=(float), snow_point_upper=(float), brightness_coeff=(float), always_apply=(bool), p=(float))
```


### 48. RandomRain
비오는 효과를 추가합니다.
```python
class albumentations.augmentations.transforms.RandomRain(slant_lower=[-20~20], slant_upper=[-20~20], drop_length=[0~100], drop_width=[1~5], drop_color=(list of (r,g,b)), blur_value=(int), brightness_coefficient=(float), rain_type=[None, "drizzle", "heavy", "torrestial"], always_apply=(bool), p=(float))
```

### 49. RandomFog
안개 효과를 추가합니다.
```python
class albumentations.augmentations.transforms.RandomFog(fog_coef_lower=(float), fog_coef_upper=(float), alpha_coef=(float), always_apply=(bool), p=(float))
```

### 50. RandomSunFlare

```python
class albumentations.augmentations.transforms.RandomSunFlare(flare_roi=(float, float, float, float), angle_lower=(float), angle_upper=(float), num_flare_circles_lower=(int), num_flare_circles_upper=(int), src_radius=(int), src_color((int, int, int)), always_apply=(bool), p=(float))
```

### 51. RandomShadow
```python
class albumentations.augmentations.transforms.RandomShadow(shadow_roi=(float, float, float, float), num_shadows_lower=(int), num_shadows_upper=(int), shadow_dimension=(int), always_apply=(bool), p=(float))
```

### 52. Lambda
**kyargs에 직접 transformation할 기법을 정해서 함수를 정의합니다.
```python
class albumentations.augmentations.transforms.Lambda(image=(callable), mask=(callable), keypoint=(callable), bbox(callable), always_apply=(bool), p=(float))
```

### 53. ChannelDropout
임의의 채널을 Drop합니다.
```python
class albumentations.augmentations.transforms.ChannelDropout(channel_drop_range=(int, int), fill_value=(int, float), always_apply=(bool), p=(float))
```

### 54. ISONoise
카메라 센서 노이즈를 적용합니다.
```python
class albumentations.augmentations.transforms.ISONoise(color_shift(float, float), intensity=((float, float)), always_apply=(bool), p=(float))
```

### 55. Solarize
임계값을 넘는 모든 화소값을 반전시킵니다.
```python
class albumentations.augmentations.transforms.Solarize(threshold=((int, int) or int, or (float, float) or float), always_apply=(bool), p=(float))
```

### 56. Equalize
이미지 히스토그램을 평탄화합니다.
```python
class albumentations.augmentations.transforms.Equalize(mode=(str), by_channel=(bool), mask=(np.ndarrary, callable), mask_params=(list of str), always_apply=(bool), p=(float))
```

### 57. Posterize
각각의 컬러 채널에 대해 num_bits만큼 감소시킵니다.
```python
class albumentations.augmentations.transforms.Posterize(num_bits((int, int) or list of ints[r,g,b]), always_apply=(bool), p=(float))
```

### 58. Downscale
이미지의 질을 다운스케일링을 통해 감소시킵니다.
```python
class albumentations.augmentations.transforms.Downscale(scale_min=(float), scale_max=(float), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```

### 59. MultiplicativeNoise
임의의 숫자 또는 배열을 이미지에 합성시킵니다.
```python
class albumentations.augmentations.transforms.MultiplicativeNoise(multiplier=(float or tuple of floats), per_channel=(bool), elementwise=(bool), always_apply=(bool), p=(float))
```

### 60. FancyPCA
RGB이미지를 “ImageNet Classification with Deep Convolutional Neural Networks”에서 사용한 FancyPCA기법을 통해 augmentation을 수행합니다.
```python
class albumentations.augmentations.transforms.FancyPCA(alpha=(float), always_apply=(bool), p=(float))
```
