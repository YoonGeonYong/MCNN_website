from flask import Blueprint, Response, request, render_template
import os
import cv2
import numpy as np
import torch

# app factory에서 db 만들고 불러오기
from influxdb_client_3 import InfluxDBClient3, Point
host = os.environ.get("INFLUXDB_HOST")
token = os.environ.get("INFLUXDB_TOKEN")
org = os.environ.get("INFLUXDB_TOKEN")

client = InfluxDBClient3(host=host, token=token, org=org)
database="density"
##

from model import MCNN

# 모델 불러오기
model = MCNN(3e-4)
model.load_state_dict(torch.load('mcnn/mcnn_trained.pth'))

cap = cv2.VideoCapture(0)

bp = Blueprint('video', __name__, url_prefix='/video')

@bp.route('/stream_code')
def show_stream_code():
    return render_template('video/stream_code.html')

@bp.route('/stream')
def send_stream():
    def generate_frames():
        while True:
            ret, frame = cap.read()  # read frame
            if not ret:
                break
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                _, encoded = cv2.imencode('.jpg', gray, params=[cv2.IMWRITE_JPEG_QUALITY, 20])
                byted = encoded.tobytes()

                ############## 모델 적용 -> imwrite ###########
                # 내부적으로 저장하고, 영상은 그냥 보여주기
                img_arr = np.fromstring(byted, np.int8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
                # model 적용
                img_f = img.astype(np.float32) / 255.0
                img_f = torch.from_numpy(img_f).unsqueeze(0)
                out_tens = model(img_f)
                out_tens = out_tens.detach().squeeze(0)
                out_arr = out_tens.cpu().numpy()

                theta = 30
                out_img = out_arr * 255 * theta
                cv2.imwrite('upload/v_output.jpg', out_img)
                ##########################################
                ########### db ##############
                tmp1 = int(np.max(out_img))
                tmp2 = int(np.min(out_img))

                client.write(database=database, record=Point('test').tag('max', tmp1).field('min', tmp2))
                ##################################
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + byted + b'\r\n') # return frame
            
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/list')
def show_list():
    return render_template('video/list.html')

@bp.route('/<int:id>/cam')
def show_camera(id):
    return render_template('video/camera.html', id=id, v_img='v_output.jpg')

@bp.route('/<int:id>/stat')
def show_statistic(id):
    # data select
    query = """SELECT *
    FROM 'test'
    WHERE time >= now() - interval '1 hours'
    AND ('bees' IS NOT NULL OR 'ants' IS NOT NULL)"""

    # Execute the query
    table = client.query(query=query, database="density", language='sql')

    # Convert to dataframe
    df = table.to_pandas().sort_values(by="time")
    print(df)
    return render_template('video/statistic.html', id=id)