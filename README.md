<<<<<<< Updated upstream
pip install influxdb3-python
=======
# GCP 배포
1. vm instance 생성 (선택: ip 고정)
2. firewall 설정
3. ssh 접속 설정 -> 접속
4. 환경 구성 (필요 패키지 설치)
5. systemd 설정

## GCP, vm instance 생성
```
Compute Engine - CREATE INSTANCE
```
ip 고정
```
VPC network -> IP addresses - RESERVE EXTERNAL STATIC ADDRESS
```

## GCP, firewall 설정
```
VPC network -> Firewall - CREATE FIREWALL RULE
```

## GCP, ssh 접속 설정
ssh key (로컬)
```sh
ssh-keygen -t rsa -f ~/.ssh/[KEY_FILENAME] -C [USERNAME]    # key 생성
cat ~/.ssh/[KEY_FILENAME].pub                               # key 확인
```
ssh key 등록 (GCP)
```
Compute Engine -> Settings -> Metadata -> SSH KEYS - ADD SSH KEY
```
ssh 접속 (로컬)
```sh
ssh -i ~/.ssh/[KEY_FILENAME] [USERNAME]@[IP]
```

## GCP, 환경 구성
### install conda
```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
```

### install packages
```sh
conda create -n crowded python=3.9
conda activate crowded

conda install flask matplotlib pytorch opencv pandas
pip install flask_socketio influxdb3-python
```

## systemd 설정
### run.sh 생성
```sh
vim /home/juchan/MCNN_website/run.sh
```
```sh
#!/bin/bash
source /home/juchan/miniconda3/etc/profile.d/conda.sh # conda
conda activate crowded

export INFLUXDB_TOKEN=".." # DB 토큰
python /home/juchan/MCNN_website/app.py # 웹 서버 실행
```
권한 부여
```sh
chmod +x /home/juchan/MCNN_website/run.sh
```

### systemd 생성
```sh
sudo vim /etc/systemd/system/crowded.service
```
```sh
[Unit]
Description=A Crowd Counting Web site
After=network.target

[Service]
User=juchan
Group=juchan
WorkingDirectory=/home/juchan/MCNN_website
ExecStart=/bin/bash /home/juchan/MCNN_website/run.sh
Environment="PATH=/home/juchan/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Restart=always

[Install]
WantedBy=multi-user.target
```

### systemd 시작
```sh
sudo systemctl daemon-reload    # 새로운 파일 인식 위한 리로드
sudo systemctl start crowded    # 서비스 시작
sudo systemctl enable crowded   # 부팅시 자동 시작

sudo systemctl status crowded   # 실행중인지 확인
```


# 할일
## GCP 사용시, 카메라 사용 불가.. -> https 필요 -> WSGI, CGI 필요
## image 처리시에 저장하지 않고, plt로 데이터 받아서 하기 (video 전송과 동일한 방식)

test
>>>>>>> Stashed changes
