from flask import Blueprint, Response, request, render_template, current_app, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import torch
from influxdb_client_3 import Point
import threading
import database
import model as m
from datetime import datetime, timedelta, timezone


bp = Blueprint('video', __name__, url_prefix='/video')


# cap = cv2.VideoCapture(0)

@bp.route('/stream_code')
def show_stream_code():
    return render_template('video/stream_code.html')

# @bp.route('/stream')
# def send_stream():
#     model = m.get_model(current_app)
#     db = database.get_db(current_app)

#     def generate_frames():
#         while True:
#             ret, frame = cap.read()  # read frame
#             if not ret:
#                 break
#             else:
#                 # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#                 _, encoded = cv2.imencode('.jpg', frame, params=[cv2.IMWRITE_JPEG_QUALITY, 20])
#                 byted = encoded.tobytes()

#                 cnt, den = counting(model, frame)
#                 inserting(db, cnt, den)
#                 # img_arr = np.fromstring(byted, np.int8)
#                 # img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
                
                
#                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + byted + b'\r\n') # return frame
            
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def counting(model, img):
#     dm = model.density_map(img)
#     x, y = model.density_point(dm)
#     den = model.density(dm)

#     return len(x), den

# def inserting(db, count, density):
#     point = Point('crowd_density').tag('id', 'galmel').field('count', count).field('density', density)
#     db.write(record=point)


@bp.route('/list')
def show_list():
    return render_template('video/list.html')

@bp.route('/<int:id>/cam')
def show_camera(id):
    return render_template('video/camera.html')

@bp.route('/statistics')
def statistic():
    return render_template('video/statistics.html')

def convert_to_kst_and_format(utc_time):
    utc_time = utc_time.split('.')[0]  # Remove microseconds
    utc_dt = datetime.strptime(utc_time, '%Y-%m-%d %H:%M:%S')
    kst_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=9)))
    return kst_dt.strftime('%Y%m.%d.%H.%M')

@bp.route('/stat')
def get_data():
    db = database.get_db(current_app)
    
    query = """SELECT *
    FROM crowd_density
    WHERE time >= now() - INTERVAL '1 hour'
    AND (count IS NOT NULL OR density IS NOT NULL)"""

    table = db.query(query=query, language='sql')
    df = table.to_pandas()  # pd.DataFrame 변환
    
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format
    df = df.sort_values(by="time", ascending=True)  # Sort by 'time' column in ascending order
    
    data = {
        'count': df['count'].tolist(),
        'density': df['density'].tolist(),
        'id': df['id'].tolist(),
        'timestamp': df['time'].astype(str).tolist()
    }
    kst_timestamps = [convert_to_kst_and_format(ts) for ts in data['timestamp']]
    modified_data = {
        "count": data["count"],
        "density": data["density"],
        "id": data["id"],
        "timestamp": kst_timestamps
    }
    return jsonify(modified_data)