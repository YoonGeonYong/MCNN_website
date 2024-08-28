import os
import time
import threading
from flask import current_app
from influxdb_client_3 import InfluxDBClient3, Point

stop_event = threading.Event()
global_data = {
    'count': 0,
    'density': 0.0
}

def init_db(app):
    host = 'https://us-east-1-1.aws.cloud2.influxdata.com'
    token = os.environ.get("INFLUXDB_TOKEN")
    org = 'Dev team'
    database = "crowded"

    print('load database connection...')
    app.config['db'] = InfluxDBClient3(host=host, token=token, org=org, database=database)
    app.config['db_thread'] = None

def get_db(app):
    return app.config['db']

# 데이터 저장
# def insert_data(app, count, density):
#     db = get_db(app)
#     point = Point('crowd_density').tag('id', 'galmel').field('count', count).field('density', density)
#     db.write(record=point)

def save_to_db(app):
    while not stop_event.is_set():
        print('is working--------')
        db = get_db(current_app)
        point = Point('crowd_density').tag('id', 'galmel').field('count', global_data['count']).field('density', global_data['density'])
        db.write(record=point)
        time.sleep(5)

def start_db_thread(app):
    if app.config['db_thread'] is None or not app.config['db_thread'].is_alive():
        stop_event.clear()
        db_thread = threading.Thread(target=save_to_db, args=(app,), daemon=True)
        db_thread.start()
        app.config['db_thread'] = db_thread
        print("DB thread started")

def stop_db_thread(app):
    stop_event.set()
    if app.config['db_thread'] is not None:
        app.config['db_thread'].join()
        app.config['db_thread'] = None
        print("DB thread stopped")
