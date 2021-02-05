---
title: Centos 네트워크 설정
categories:
- OS
last_modified_at: 2021-02-01T14:00:00+09:00
tags:
- OS
- LINUX
- VBox

toc: true
---
# 1. Intro
리눅스에서 wget모듈이나 서버의 역할을 수행하기 위해서는 네트워크 설정이 필수적이여서 어댑터 브리지 방식으로
네트워크를 설정하는 방식을 정리합니다. NAT방식과 어댑터 브리지 방식의 차이점은 다른 포스트를 통해 소개할 예정입니다.

# 2. 네트워크 설정
#### 2.1. 기본 설정
![Centos8](https://user-images.githubusercontent.com/56510688/106564038-0506b400-6570-11eb-958c-e51134a0c25e.PNG)
VM 가상 OS 환경설정 네트워크탭에서 어댑터 브리지 방식으로 바꿔준 후 OS를 부팅합니다. 그 후 커널에서
```
{
# ip addr show
}
```
를 통해서 ip를 확인합니다. ipconfig는 wget을 통해 설치하면 사용할 수 있지만 디폴트로 안깔려있어 사용할 수 없습니다.

#### 2.3. NIC device 설정 파일 열기
ip addr show 명령어를 통해 NIC이름을 파악했으면 해당 설정 파일을 열어 내용을 수정해야합니다.
```
{
<!--Root계정-->
# vi /etc/sysconfig/network-scripts/ifcfg-enp0s3
<!--일반계정-->
# sudo vi /etc/sysconfig/network-scripts/ifcfg-enp0s3
}
```
를 통해 파일을 열어 내용을 확인합니다. 만약 로그인한 계정이 루트 계정이 아닐 시에는 파일 수정 권한이
없어 열람만 할 수 있으므로 두번째 명령어를 사용해 관리자 권한으로 실행하면됩니다.

#### 2.4. NIC device 설정 파일 수정하기
![network1](https://user-images.githubusercontent.com/56510688/106846851-090e0f80-66f1-11eb-95ec-9e4e3f38611d.PNG)
파일을 열면 기본적으로 설정되있는 내용이 있습니다. 그 후 아래내용을 수정, 추가합니다.
ip의 경우에는 호스트 OS에서 ipconfig를 통해 ip를 확인한 후 같은 대역의 주소를 할당하면됩니다.
```
{
BOOTPROTO=static
ONBOOT=yes
IPADDR=220.69.208.xxx
NETMASK=255.255.255.0
GATEWAY=220.69.208.1
DNS1=8.8.8.8
DNS2=8.8.6.6
}
```

#### 2.5. 네트워크 연결 확인하기
```
{
# service network restart
# ping 8.8.8.8
}
```
네트워크를 재시작한 후 핑 명령어를 통해 네트워크가 연결이 되있는지 확인합니다.
![network2](https://user-images.githubusercontent.com/56510688/106846854-0a3f3c80-66f1-11eb-803b-d956ec3c2c09.PNG)
제대로 네트워크 연결이 되어있지 않다면 다음과 같이 unreachable이라는 상태 메세지와 Packet loss 100%가 뜹니다.

![network3](https://user-images.githubusercontent.com/56510688/106846855-0a3f3c80-66f1-11eb-95dc-3dd52c4ded4b.PNG)
제대로 연결되어있다면 전송 지연속도를 나타내며 정상적으로 패킷이 전송되었다고 알려줍니다.

# 3. 맞닥뜨린 문제
가상 OS를 밀어버린 후 다시 설치하였을때 NIC device가 비활성화된 경우가 있는데 아래 명령어를 통해 해결하였습니다.
```
{
# nmcli con (NIC명)
}
```

