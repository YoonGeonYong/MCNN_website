from flask import Flask, current_app, render_template
from flask_socketio import SocketIO
import database
import model
import events
from views import main_views, info_views, video_views, image_views
import matplotlib

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

# socket 이벤트 핸들러 등록
socketio = events.init_socketio(app)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True) # 
