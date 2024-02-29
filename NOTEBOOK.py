# Flask 라이브러리 import
from flask import Flask, render_template, request, send_from_directory

# OS, OpenCV, Numpy 라이브러리 import
import os
import cv2
import numpy as np

# PyTorch 라이브러리 import
import torch

# matplotlib 라이브러리 import
import matplotlib
from matplotlib import pyplot as plt

# 모델 불러오기 (Conv2d, MCNN 클래스 정의되어 있어야 함)
from model import Conv2d, MCNN

# 데이터셋 클래스 및 데이터 증강 함수 불러오기 (MyDataset 클래스, aug_train, aug_val 함수 정의되어 있어야 함)
from MYDataset import MyDataset, aug_train, aug_val

# train-test split 함수 불러오기
from sklearn.model_selection import train_test_split

# Flask 앱 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # 업로드 폴더 설정

# 파일 업로드 처리 함수
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 업로드된 파일 처리
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
        file.save(filename)

        # 업로드된 이미지 읽기
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # BGR -> 그레이스케일 변환

        # 이미지를 float형으로 변환하고 0~1 사이의 값으로 정규화
        im = im.astype(np.float32) / 255.0

        # 이미지를 PyTorch 텐서로 변환하고 배치 차원 추가
        im_tensor = torch.from_numpy(im).unsqueeze(0)

        # 모델 불러오기
        model = MCNN(3e-4)
        model.load_state_dict(torch.load('mcnn_model.pth'))

        # 모델 예측
        output = model(im_tensor)

        # 예측 결과 텐서 분리 및 배치 차원 제거
        output_image = output.detach().squeeze(0)

        # --- 웹페이지에 시각화 결과 출력하기 위한 코드 추가 ---
        plt.figure(figsize=(6, 6))
        plt.imshow(output_image_np)  # 예측 결과 이미지 출력 (output_image_np 변수는 아직 정의되지 않았음)
        plt.title('모델 예측 결과')

        # 시각화 결과를 임시 이미지 파일로 저장 (실제 웹 환경에서는 필요하지 않음)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'model_output.png'))

        # 업로드 이미지와 예측 결과 이미지를 함께 템플릿에 전달
        return render_template('index.html', filename1='uploaded_image.png', filename2='temp_output.png')

    # 파일 업로드 화면 제공
    return render_template('index.html')

# 업로드된 파일 반환 함수
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 프로그램 실행 (flask 실행)
if __name__ == '__main__':
    app.run(debug=True)
