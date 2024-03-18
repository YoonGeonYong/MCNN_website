from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
plt.switch_backend('Agg')  # 'Agg' 백엔드를 사용하여 GUI 없이도 그래프를 저장할 수 있음

from model import MCNN  # Conv2d는 사용하지 않으므로 제외
from MYDataset import MyDataset, aug_train, aug_val  # 사용하지 않는 경우 제거할 것
from sklearn.model_selection import train_test_split  # 사용하지 않는 경우 제거할 것

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
        file.save(filename)
        
        im = cv2.imread(filename)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = im_gray.astype(np.float32) / 255.0

        # 채널 차원 추가 및 배치 차원 추가
        im_tensor = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)

        model = MCNN(3e-4)
        
        state_dict = torch.load('server_model_weights_round_1.pth')
        if isinstance(state_dict, tuple):
            # 튜플의 첫 번째 요소가 상태 딕셔너리라고 가정
            state_dict = state_dict[0]

        model.load_state_dict(state_dict)
        
        output = model(im_tensor)

        output_image = output.detach().squeeze(0).squeeze(0)  # 배치 차원 및 채널 차원 제거

        plt.figure(figsize=(6, 6))
        plt.imshow(output_image.cpu().numpy(), cmap='gray')  # 그레이스케일 이미지로 시각화
        plt.title('Model Output')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'model_output.png'))

        return render_template('index.html', filename1='uploaded_image.png', filename2='model_output.png')

    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
