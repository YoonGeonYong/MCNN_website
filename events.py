from io import BytesIO
import cv2
from flask_socketio import SocketIO, emit
from influxdb_client_3 import Point
from matplotlib import pyplot as plt
import numpy as np
import base64

    
def init_socketio(app):
    socketio = SocketIO(app)
    app.config['socketio'] = socketio

    model = app.config['model']
    db = app.config['db']

    # 이벤트 핸들러 등록
    # image 수신 및 처리
    @socketio.on('image')
    def handle_image(data):
        img = base64.b64decode(data['image'])
        id = data['id']

        '''img 처리'''
        # decoding
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

        # model 처리
        dm = model.density_map(img)
        x, y = model.density_point(dm)
        den = model.density(dm)

        # 시각화
        fig, axes = plt.subplots(1, 2, constrained_layout=True)

        axes[0].set_title('Density Map')
        axes[0].set_axis_off()
        axes[0].imshow(dm, cmap='jet')

        axes[1].set_title(f'Density Point (Count: {len(y)})')
        axes[1].set_axis_off()
        axes[1].imshow(img, cmap='gray')
        axes[1].scatter(x * 4, y * 4, color='r', s=5)

        buf = BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0.1) # bbox_inches='tight', pad_inches=0.1 여백 최소화
        buf.seek(0)
        plt.close(fig)


        '''img 전송'''
        # encoding
        _img = base64.b64encode(buf.getvalue()).decode('utf-8')
        emit('response', {
            'image': _img,
            'count': len(y),
            'density': den
        })

        '''db 저장'''
        point = Point('crowd_density').tag('id', id).field('count', len(x)).field('density', den)
        db.write(record=point)

    return socketio