---
title: Anaconda 3에서 가상환경 구축하기
categories:
- Anaconda 3
last_modified_at: 2020-08-22T14:00:00+09:00
tags:
- Anaconda 3
toc: true
---

## 1. Anaconda 3 기본 명령어
__해당 환경에 설치된 패키지 리스트 불러오기__
```
conda list
```
__패키지 설치(site-packages 폴더에 저장)__
```
conda install [패키지 이름]
```
__패키지 업데이트__
```
conda update [패키지 이름]
```
__패키지 제거__
```
conda remove [패키지 이름]
```
__패키지 검색__
```
conda search '*[패키지 이름]*'
```

## 2. Anaconda 3 가상환경
가상환경을 구축하기 위해서 아래 두가지 방법 중 하나로 Anaconda Prompt(CMD)를 실행시켜 줍니다.

__1. 파일검색기 이용하기__

<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90956373-fe541b00-e4c0-11ea-9fc4-5e45232de6b5.png"></p>

__2. Anaconda Navigator 이용하기__

<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90956374-00b67500-e4c1-11ea-8a05-19839bd9ec30.JPG"></p>

__가상환경 만들기__

python=[파이썬 버젼]은 생략해도 되고  default값은 최신버젼 입니다.
```
conda create -n [가상환경 이름] python=[파이썬 버젼]
```
가상환경을 처음 만들면 아무런 패키지가 없어서 기본적으로 Anaconda 3가 제공해주는 패키지를 다운 받을지 물어봅니다.
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90956712-fc3f8b80-e4c3-11ea-96e2-67cda6fd960f.JPG")></p>

__가상환경 시작__

콘다 가상환경을 활성화 하면 아래에 이미지처럼 왼쪽에 (base)였던게 (가상환경 이름)으로 바뀝니다.
<p align="center"><img src="https://user-images.githubusercontent.com/56510688/90956714-006ba900-e4c4-11ea-8b83-477b75936234.JPG"></p>
```
conda activate [가상환경 이름]
```
__가상환경 종료__

가상환경 안에 있는 상태로 명령어를 실행해야합니다.
```
conda deactivate
```
__가상환경 저장하기__

yaml파일로 가상환경을 저장할 수 있습니다.
```
conda env export > [가상환경 이름].yaml
```
__가상환경 불러오기__

yaml파일의 가상환경 이름으로 가상환경을 불러오는 명령어라 yaml파일과 같은 이름의 가상환경이 이미 존재하면 생성이 안됩니다.
```
conda env create -f [가상환경 이름].yaml
```
__가상환경 제거__
```
conda env remove -n [가상환경 이름]
```
## 3. Anaconda 3 가상환경 jupyter notebook과 연결하기
__kernel 추가__

가상환경을 activate한 상태에서 아래 명령어를 통해 jupyter notebook을 설치한다.
```
pip install ipykernel
```
그리고 아래 명령어를 통해 kernel과 추가하면 된다.
```
python -m ipykernel install --user --name [가상환경 이름] --display-name "[jupyter에서 보여질 이름]"
```
__kernel 제거__

명령어에 의해 추가된 가상환경을 삭제해도 kernel은 유지되므로 삭제하고 싶으면 아래 명령어를 통해 삭제하면된다.
```
jupyter kernelspec uninstall [가상환경이름]
```
