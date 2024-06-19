import matplotlib
matplotlib.use('agg') # NSInternalInconsistencyException : NSWindow should only be instantiated on the main thread -> 방지
from flask import Flask

app = Flask(__name__) # flask app

# 객체 초기화
import database, model, events
database.init_db(app)                           # DB
model.init_model(app, './mcnn/trained_B.pth')   # 모델

# Blueprint 등록
from views import main_views, info_views, video_views, image_views
app.register_blueprint(main_views.bp)
app.register_blueprint(info_views.bp)
app.register_blueprint(video_views.bp)
app.register_blueprint(image_views.bp)

socketio = events.init_socketio(app) # socketio : 웹 소켓 (+ 이벤트 핸들러 등록)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)