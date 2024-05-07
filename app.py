from flask import Flask

# application factory
def create_app():
    app = Flask(__name__)
    app.debug = True

    # DB

    # blueprint
    from views import main_views, info_views, video_views, image_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(info_views.bp)
    app.register_blueprint(video_views.bp)
    app.register_blueprint(image_views.bp)

    return app