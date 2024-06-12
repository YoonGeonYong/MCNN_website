from flask import Flask, current_app, render_template
from flask_socketio import SocketIO, emit
import database
import model
from views import main_views, info_views, video_views, image_views
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
from influxdb_client_3 import Point

matplotlib.use('agg')

app = Flask(__name__)

# DB 초기화
database.init_db(app)

# 모델 초기화
model.init_model(app, './mcnn/trained_B.pth')

# Blueprint 등록
app.register_blueprint(main_views.bp)
app.register_blueprint(info_views.bp)
app.register_blueprint(video_views.bp)
app.register_blueprint(image_views.bp)

socketio = SocketIO(app)
model = app.config['model']

# 전역 변수로 count와 density를 저장
global_data = database.global_data

@socketio.on('start_stream')
def start_stream():
    database.start_db_thread(current_app)

@socketio.on('stop_stream')
def stop_stream():
    database.stop_db_thread(current_app)

@socketio.on('image')
def handle_image(data):
    # Decode the image from base64
    image_data = base64.b64decode(data)
    npimg = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    # Generate density map and points
    dm = model.density_map(img)
    x, y = model.density_point(dm)
    den = model.density(dm)

    # Visualization
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    axes[0].set_title('Density Map')
    axes[0].set_axis_off()
    axes[0].imshow(dm, cmap='jet')

    axes[1].set_title(f'Density Point (Count: {len(y)})')
    axes[1].set_axis_off()
    axes[1].imshow(img, cmap='gray')
    axes[1].scatter(x * 4, y * 4, color='r', s=5)

    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0.1) # bbox_inches='tight', pad_inches=0.1 여백 최소화
    buf.seek(0)
    plt.close(fig)

    # Encode image to base64 to send back to the client
    output_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    emit('response', {
        'image': output_img,
        'count': len(y),
        'density': den
    })

    # Update global data
    global_data['count'] = len(y)
    global_data['density'] = den

    db = database.get_db(app)
    point = Point('crowd_density').tag('id', 'galmel').field('count', len(x)).field('density', den)
    db.write(record=point)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True)
