# GCP, ssh 접속
ssh key (로컬)
```sh
ssh-keygen -t rsa -f ~/.ssh/[KEY_FILENAME] -C [USERNAME]    # key 생성
cat ~/.ssh/[KEY_FILENAME].pub                               # key 확인
```
ssh key 등록 (GCP)
```
Compute Engine -> Settings -> Metadata -> SSH keys -> add ssh key에 복사
```
ssh 접속 (로컬)
```sh
ssh -i [PRIVATE KEY] [USERNAME]@[외부IP]
```

# conda install
```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
```

# packages
```sh
conda create -n crowded python=3.9
conda activate crowded

conda install flask matplotlib pytorch opencv pandas
pip install flask_socketio influxdb3-python
```