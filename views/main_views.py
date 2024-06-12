from flask import Blueprint, render_template, current_app

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('index.html') # templates 폴더를 기본으로 인식

@bp.route('/el')
def show_element():
    return render_template('elements.html')

# DB test
@bp.route('/db/<int:n1>/<int:n2>')
def test_db(n1, n2):
    from influxdb_client_3 import Point
    import database

    db = database.get_db(current_app)

    # 데이터 저장
    point = Point('test').tag('max', n1).field('min', n2)
    db.write(record=point)

    # 데이터 검색
    query = """SELECT * FROM 'test' WHERE time >= now() - interval '1 hours' AND ('max' IS NOT NULL OR 'min' IS NOT NULL)"""
    table = db.query(query=query, language='sql')
    df = table.to_pandas().sort_values(by="time", ascending=False)			# pd.DataFrame 변환		/ to_pylist() : 리스트로 변환
    data = df.iloc[0,:].astype('string').to_list()

    return data

# model test
@bp.route('model/<int:num>')
def test_model(num):
    import model as m
    import cv2

    model = m.get_model(current_app)

    # 이미지 읽기
    img_path = '~/workspace/projects/CrowdCounting/MCNN_svishwa/data/original/shanghaitech/part_A_final/test_data/images/IMG_'+ str(num) +'.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 모델 처리
    dm = model.density_map(img)
    den = model.density(dm)

    return str(den)