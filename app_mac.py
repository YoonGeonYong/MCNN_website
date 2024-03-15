from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

from model import Conv2d, MCNN
from MYDataset import MyDataset, aug_train, aug_val
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'    # 업로드 폴더 설정

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 업로드된 파일 처리
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
        file.save(filename)
        
        # 업로드된 이미지 읽기
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # BGR -> 그레이스케일 변환

        # 이미지를 float형으로 변환하고 0~1 사이의 값으로 정규화
        im = im.astype(np.float32) / 255.0

        # 이미지를 PyTorch 텐서로 변환하고 배치 차원 추가
        im_tensor = torch.from_numpy(im).unsqueeze(0)   # 배치 차원 추가 (가장 앞에 차원 추가)

        # 모델 불러오기
        model = MCNN(3e-4)
        # model.load_state_dict(torch.load('mcnn_model.pth'))
        model.load_state_dict(torch.load('server_model_weights_round_1.pth'))
        
        
        
        
        # 모델 예측
        output = model(im_tensor)

        # 예측 결과 텐서 분리 및 배치 차원 제거
        output_image = output.detach().squeeze(0)

        # 예측 결과 시각화
        output_image_np = output_image.cpu().numpy()    # 텐서 -> 넘파이 배열 변환
        plt.figure(figsize=(6, 6))
        plt.imshow(output_image_np)
        plt.title('Model Output')

        # 시각화 결과 저장
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'model_output.png'))

        # 업로드 이미지와 예측 결과 이미지를 함께 템플릿에 전달
        return render_template('index.html', filename1='uploaded_image.png', filename2='model_output.png')

    # 파일 업로드 화면 제공
    return render_template('index.html')


# 업로드된 파일 반환 함수
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 프로그램 실행 (flask 실행)
if __name__ == '__main__':
    app.run(debug=True)
