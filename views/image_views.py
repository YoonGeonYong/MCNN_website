from flask import Blueprint, request, render_template, send_from_directory, make_response, Response, current_app
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import model as m

bp = Blueprint('image', __name__, url_prefix='/image')

@bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # 원본 파일
        img_byte = request.files['file'].read()  # bytes
        if len(img_byte) == 0:  # img가 없을경우 다시 입력화면으로
            return render_template('image/image.html')
        
        img_arr = np.frombuffer(img_byte, np.int8)  # bytes -> ndarray
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)  # cv2 변환

        cv2.imwrite('upload/input.jpg', img)  # 원본 저장

        # model 적용
        model = m.get_model(current_app)
        dm = model.density_map(img)
        x, y = model.density_point(dm)
        den = model.density(dm)

        # matplotlib를 사용하여 이미지 저장
        plt.imshow(dm)
        plt.axis('off')

        # 이미지 저장을 위한 버퍼 생성
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        # 버퍼 내용을 파일로 저장
        with open('upload/output.jpg', 'wb') as f:
            f.write(buf.getvalue())

        buf.close()

        return render_template('image/image.html', in_img='input.jpg', out_img='output.jpg', num=len(x), density=den)

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