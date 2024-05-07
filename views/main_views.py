from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('index.html') # templates 폴더를 기본으로 인식

@bp.route('/el')
def show_element():
    return render_template('elements.html')