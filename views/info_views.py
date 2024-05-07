from flask import Blueprint, render_template

bp = Blueprint('info', __name__, url_prefix='/info')

@bp.route('/web')
def intro_website():
    return render_template('info/website.html')

@bp.route('/model')
def intro_model():
    return render_template('info/model.html')

@bp.route('/dev')
def intro_developers():
    return render_template('info/developers.html')