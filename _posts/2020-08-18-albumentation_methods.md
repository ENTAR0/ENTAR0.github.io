---
title: albuementation method 정리
categories:
- Image augmentation
last_modified_at: 2020-08-19T14:00:00+09:00
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
- always_apply=(Boolean) : Default값은 False로 설정되있는데 딱히 신경안써도 된다.
- p=(float) : 해당 augmentation option을 몇퍼센트의 확률로 적용할지 결정한다. (ex. p=0.5면 50%확률)
- interpolation=(OpenCV flag) : cv2라이브러리에서 제공하는 보간법(interpolation)을 적용한다.
  - OpenCV flag=0 : cv2.INTER_NEAREST 적용
  - OpenCV flag=1 : cv2.INTER_LINEAR 적용
  - OpenCV flag=2 : cv2.INTER_CUBIC 적용
  - OpenCV flag=3 : cv2.INTER_AREA 적용
  - OpenCV flag=4 : cv2.INTER_LANCZOS4 적용
- border_mode=(OpenCV flag) : cv2라이브러리에서 제공하는 외간법(extrapolation)을 적용한다
  - OpenCV flag=0 : cv2.BORDER_CONSTANT 적용
  - OpenCV flag=1 : cv2.BORDER_REPLICATE 적용 - 추천
  - OpenCV flag=2 : cv2.BORDER_REFLECT 적용
  - OpenCV flag=3 : cv2.BORDER_WRAP 적용
  - OpenCV flag=4 : cv2.BORDER_REFLECT_101 적용


# 3. albumentation methods(작성중)
#### 1. Blur
```python
class albumentations.augmentations.transforms.Blur(blur_limit(int, (int, int)), always_apply=(bool), p=(float))
```


#### 2. VerticalFlip
```python
class albumentations.augmentations.transforms.VerticalFlip(always_apply=(bool), p=(float))
```


#### 3. HorizontalFlip
```python
class albumentations.augmentations.transforms.HorizontalFlip(always_apply=(bool), p=(float))
```


#### 4. Flip
```python
class albumentations.augmentations.transforms.Flip(always_apply=(bool), p=(float))
```


#### 5. Normalize
```python
class albumentations.augmentations.transforms.Normalize(mean=(float, list of float), std=(float, list of float), max_pixel_value=(float),
always_apply=(bool), p=(float))
```


#### 6. Transpose
```python
class albumentations.augmentations.transforms.Transpose(always_apply=(bool), p=(float))
```


#### 7. RandomCrop
```python
class albumentations.augmentations.transforms.RandomCrop(height=(int), width=(int), always_apply=(bool), p=(float))
```


#### 8. RandomGamma
```python
class albumentations.augmentations.transforms.RandomGamma(gamma_limit=(float or (float, float)), eps=None, always_apply=(bool), p=(float))
```


#### 9. RandomRotate90
```python
class albumentations.augmentations.transforms.RandomRotate90(always_apply=(bool), p=(float))
```


#### 10. Rotate
```python
class albumentations.augmentations.transforms.Rotate(limit=((int, int) or int), interpolation=(OpenCV flag), border_mode=(OpenCV flag),
value=(int, float, list of ints, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 11. ShiftScaleRotate
```python
class albumentations.augmentations.transforms.ShiftScaleRotate(shift_limit=((float, float) or float), scale_limit((float, float) or float), rotate_limit((int, int or int),
interpolation=(OpenCV flag), border_mode(OpenCV flag), value(int, float, list of int, list of float), mask_value=(int, float),
always_apply=(bool), p=(float))
```


#### 12. CenterCrop
```python
class albumentations.augmentations.transforms.CenterCrop(height=(int), width=(int), always_apply=(bool), p=(float))
```


#### 13. OpticalDistortion
```python
class albumentations.augmentations.transforms.OpticalDistortion(distort_limit=(float, (float, float)), shift_limit=(float, (float, float)),
interpolation=(OpenCV flag), border_mode=(OpenCV flag), value=(int, float, list of ints, list of float),
mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 14. GridDistortion
```python
class albumentations.augmentations.transforms.GridDistortion(num_steps=(int), distort_limit=(float, (float, float)), interpolation=(OpenCV flag), border_mode=(OpenCV flag),
value=(int, float, list of ints, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 15. ElasticTransform
```python
class albumentations.augmentations.transforms.ElasticTransform(alpha=(float), sigma=(float), alpha_affine=(float), interpolation=(OpenCV flag), border_mode=(OpenCV flag),
value=(int, float, list of ints, list of float), mask_value=(int, float), approximate=(bool), always_apply=(bool), p=(float))
```


#### 16. RandomGridShuffle
```python
class albumentations.augmentations.transforms.RandomGridShuffle(grid=((int, int) always_apply=(bool), p=(float))
```


#### 17. HueSaturationValue
```python
class albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=((int, int) or int), sat_shift_limit=((int, int) or int), val_shift_limit=((int, int) or int), always_apply=(bool), p=(float))
```


#### 18. PadIfNeeded
```python
class albumentations.augmentations.transforms.PadIfNeeded(min_height=(int), min_width=(int), border_mode=(OpenCV flag), value=(int, float, list of int, list of float), mask_value=(int, float), always_apply=(bool), p=(float))
```


#### 19. RGBshift
```python
class albumentations.augmentations.transforms.RGBshift(r_shift_limit=((int, int) or int), g_shift_limit=((int, int) or int), b_shift_limit=((int, int) or int), p=(float))
```


#### 20. RandomBrightness
```python
class albumentations.augmentations.transforms.RandomBrightness(limit=((float, float) or float), always_apply=(bool), p=(float))
```


#### 21. RandomContrast
```python
class albumentations.augmentations.transforms.RandomContrast(limit=((float, float) or float), always_apply=(bool), p=(float))
```


#### 22. MotionBlur
```python
class albumentations.augmentations.transforms.MotionBlur(blur_limit=(int), always_apply=(bool), p=(float))
```


#### 23. MedianBlur
```python
class albumentations.augmentations.transforms.MedianBlur(blur_limit=(int), always_apply=(bool), p=(float))
```


#### 24. GaussianBlur
```python
class albumentations.augmentations.tranforms.GaussianBlur(blur_limit=(int), always_apply=(bool), p=(float))
```


#### 25. GaussNoise
```python
class albumentations.augmentations.transforms.GaussNoise(var_limit=((float, float) or float), mean=(float), always_apply=(bool), p=(float))
```


#### 26. GlassBlur
```python
class albumentations.augmentations.transforms.GlassBlur(sigma=(float), max_delta=(int), iterations=(int), mode=(str), always_apply=(bool), p=(float))
```


#### 27. CLAHE
```python
class albumentations.augmentations.transforms.CLAHE(clip_limit=(float or (float, float)), tile_grid_size=((int, int)), always_apply=(bool), p=(float))
```


#### 28.ChannelShuffle
```python
class albumentations.augmentations.transforms.ChannelShuffle(always_apply=(bool), p=(float))
```


#### 29. InverImg
```python
class albumentations.augmentations.transforms.InvertImg(always_apply=(bool), p=(float))
```


#### 30. ToGray
```python
class albumentations.augmentations.transforms.ToGray(always_apply=(bool), p=(float))
```


#### 31. ToSepia
```python
class albumentations.augmentations.transforms.ToSepia(always_apply=(bool), p=(float))
```


#### 32. JpegCompression
```python
class albumentations.augmentations.transforms.JpegCompression(quality_lower=(float), quality_upper=(float), always_apply=(bool), p=(float))
```


#### 33. ImageCompression
```python
class albumentations.augmentations.transforms.ImageCompression(quality_lower=(float), quality_upper=(float), compression_type=(ImageCompressionType), always_apply=(bool), p=(float))
```


#### 34. Cutout
```python
class albumentations.augmentations.transforms.Cutout(num_holes=(int), max_h_size=(int), max_w_size=(int), fill_value=(int, float, list of int, list of float), always_apply=(bool), p=(float))
```


#### 35. CoarseDropout
```python
class albumentations.augmentations.transforms.CoarseDropout(max_holes=(int), max_height=(int), max_width=(int), min_holes=(int), min_height=(int), min_width=(int), fill_value=(int, float, list of int, list of float), always_apply=(bool), p=(float))
```


#### 36. ToFloat
```python
class albumentations.augmentations.transforms.ToFloat(max_value=(float), always_apply=(bool), p=(float))
```


#### 37. FromFloat
```python
class albumentations.augmentations.transforms.FromFloat(max_value=(float), always_apply=(bool), p=(float))
```


#### 38. Crop
```python
class albumentations.augmentations.transforms.Crop(x_min=(int), y_min=(int), x_max=(int), y_max=(int), always_apply=(bool), p=(float))
```


#### 39. CropNonEmptyMaskIfExists
```python
class albumentations.augmentations.transforms.CropNonEmptyMaskIfExists(height=(int), width=(int), ignore_values=(list of int), ignore_channels=(list of int), always_apply=(bool), p=(float))\
```


#### 40. RandomScale
```python
class albumentations.augmentations.transforms.RandomScale(scale_limit=((float, float), or float), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```


#### 41. SamllestaMaxSize
```python
class albumentations.augmentations.transforms.SmallestMaxSize(max_size=(int), interpolation=(OpneCV flag), always_apply=(bool), p=(float))
```


#### 42. Resize
```python
class albumentations.augmentations.transforms.Resize(height=(int), width=(int), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```


#### 43. RandomResizedCrop
```python
class albumentations.augmentations.transforms.RandomResizedCrop(height=(int), width=(int), scale=((float, float)), ratio=((float, float)), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```


#### 44. RandomBrightnessContrast
```python
class albumentations.augmentations.transforms.RandomBrightnessContrast(brightness_limit=((float, float) or float), contrast_limit=((float, float), or float), brightness_by_max=(Boolean), always_apply=(bool), p=(float))
```


#### 45. RandomCropNearBBox
```python
transforms.RandomCropNearBBox(max_part_shift=(float), always_apply=(bool), p=(float))
```


#### 46. RandomSizedBBoxSafeCrop
```python
transforms.RandomSizedBBoxSafeCrop(height=(int), width=(int), erosion_rate=(float), interpolation=(OpenCV flag), always_apply=(bool), p=(float))
```
