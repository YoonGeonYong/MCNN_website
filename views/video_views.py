from flask import Blueprint, render_template, current_app, jsonify
import pandas as pd
from datetime import datetime, timedelta, timezone
 

bp = Blueprint('video', __name__, url_prefix='/video')


@bp.route('/stream')
def show_stream():
    return render_template('video/stream.html')

@bp.route('/list')
def show_list():
    db = current_app.config['db']
    
    query = """SELECT DISTINCT id
                FROM "crowd_density"
                WHERE time >= now() - interval '1 hour'"""

    table = db.query(query=query, language='sql')
    df = table.to_pandas()  # pd.DataFrame 변환
    
    id_list = df['id'].tolist()

    return render_template('video/list.html', id_list=id_list)

@bp.route('/<string:id>/cam')
def show_camera(id):
    return render_template('video/camera.html', id=id)

@bp.route('/<string:id>/stat')
def statistic(id):
    return render_template('video/statistics.html', id=id)

def convert_to_kst_and_format(utc_time):
    utc_time = utc_time.split('.')[0]  # Remove microseconds
    utc_dt = datetime.strptime(utc_time, '%Y-%m-%d %H:%M:%S')
    kst_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=9)))
    return kst_dt.strftime('%Y%m.%d.%H.%M')

@bp.route('/<string:id>/data') # plot하기 위해서 데이터를 db에서 불러서 jsonify
def get_data(id):
    db = current_app.config['db']
    
    query = f"""SELECT *
                FROM "crowd_density"
                WHERE "id" IN ('{id}')
                AND time >= now() - interval '1 hour'"""

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