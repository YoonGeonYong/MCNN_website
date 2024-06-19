from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('index.html') # templates/ (defaults)



''' test '''
# elements sample
@bp.route('/el')
def show_element():
    return render_template('elements.html')

# DB test
@bp.route('/db/<string:id>/<int:count>/<float:density>') # str, int, float 타입 맞춰야함
def test_db(id, count, density):
    from influxdb_client_3 import Point
    from flask import current_app

    db = current_app.config['db']

    # 데이터 저장
    point = Point('crowd_density').tag('id', id).field('count', count).field('density', density)
    db.write(record=point)

    # 데이터 검색
    query = """SELECT * FROM 'crowd_density' WHERE time >= now() - interval '1 hours' AND ('count' IS NOT NULL OR 'density' IS NOT NULL)"""
    table = db.query(query=query, language='sql')
    df = table.to_pandas().sort_values(by="time", ascending=False)			# pd.DataFrame 변환		/ to_pylist() : 리스트로 변환
    data = df.iloc[0,:].astype('string').to_list()

    return data

# model test
@bp.route('/model/<int:num>')
def test_model(num):
    from flask import current_app
    import cv2

    # 이미지 읽기
    img_path = '/Users/leejuchan/workspace/projects/CrowdCounting/MCNN_svishwa/data/original/shanghaitech/part_A_final/test_data/images/IMG_'+ str(num) +'.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 모델 처리
    model = current_app.config['model']
    dm = model.density_map(img)
    den = model.density(dm)

    return str(den)