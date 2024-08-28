#!/bin/bash
source /home/juchan/miniconda3/etc/profile.d/conda.sh # conda 환경 접속
conda activate crowded

export INFLUXDB_TOKEN="CwigcmHO_Iq6misRiOIlA3cPip9jioBn2cepwwjZ17ZdusoU3rMDcY4fFLdWzWvZ5mSgC0OJTUzxkriXE7UGig==" # DB 토큰
cd /home/juchan/MCNN_website
python app.py # 웹 서버 실행