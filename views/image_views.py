from flask import Blueprint, request, render_template, send_from_directory, make_response, Response
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from mcnn.model import MCNN

# 모델 불러오기
model = MCNN(3e-4)
model.load_state_dict(torch.load('mcnn/mcnn_trained.pth'))

bp = Blueprint('image', __name__, url_prefix='/image')

@bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # 원본 파일
        img_byte = request.files['file'].read()             # bytes
        if len(img_byte) == 0:                              # img가 없을경우 다시 입력화면으로
            return render_template('image/image.html')
        img_arr = np.fromstring(img_byte, np.int8)          # bytes -> ndarray
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)   # cv2 변환

        cv2.imwrite('upload/input.jpg', img) # 원본 저장

        # model 적용
        img_f = img.astype(np.float32) / 255.0              # [0,1]로 정규화
        img_f = torch.from_numpy(img_f).unsqueeze(0)        # batch_size 차원 추가 : (h, w) -> (1, h, w)
        out_tens = model(img_f)
        out_tens = out_tens.detach().squeeze(0)             # batch_size 차원 제거 : (1, h, w) -> (h, w)
        out_arr = out_tens.cpu().numpy()                    # tensor -> ndarray
        
        theta = 17
        out_img = out_arr * 255 * theta                     # [0~255] 변환 * theta (값이 너무 작아서 안보임)
        # print(out_img)
        # print(np.max(out_img), np.min(out_img))

        cv2.imwrite('upload/output.jpg', out_img) # 결과물 저장

        return render_template('image/image.html', in_img='input.jpg', out_img='output.jpg')

    # GET : 업로드 화면
    return render_template('image/image.html')


# 업로드된 파일 반환 함수
@bp.route('/send/<filename>')
def send_file(filename):
    return send_from_directory('upload/', filename)

# @bp.route('/send')
# def send_file(img):
#     cv2.imread
#     return Response(img, mimetype='image/jpeg')


# 이미지 반환 테스트
@bp.route('/')
def show_img():
    img_path = 'lena.jpeg'
    
    # img 인코딩
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # ndarray
    _, encoded = cv2.imencode('.jpeg', img)             # jpeg (encoding)
    byted = encoded.tobytes()                           # bytes

    return Response(byted, mimetype='image/jpeg')