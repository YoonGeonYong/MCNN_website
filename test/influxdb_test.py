'''
	InfluxDB 설명 (ref : https://mangkyu.tistory.com/188, https://mangkyu.tistory.com/190)

	- TSDB (Time Series Database)
	: 실시간으로 쌓이는 대규모 데이터를 처리하기 위한 DB (for 시계열 데이터)	-> 쓰기/읽기 속도, 보존정책

	- 핵심기능
		1. Continous Query: 일정 주기마다 데이터 처리 (다운 샘플링)
		2. Retention Policy: 일정 주기마다 데이터 삭제
		3. Restful API 제공
	
	- 구조
		Database	= database
		Measurement	= table 				/ schemaless

		Key			= column
			Tag Key		= Indexed column 	/ str
			Field Key	= unindexed column	/ 1~
		Point		= row

		(Series, Shard)
'''

import os, time
from influxdb_client_3 import InfluxDBClient3, Point
import numpy as np

host = 'https://us-east-1-1.aws.cloud2.influxdata.com'
org = 'Dev team'
database="crowded"
token = os.environ.get("INFLUXDB_TOKEN")


# data insert
db = InfluxDBClient3(host=host, org=org, database=database, token=token)
point = Point('crowd_density').tag('id', 'galmel').field('count', 3).field('density', 10.5)
db.write(record=point)

# data select
query = """SELECT *
FROM 'crowd_density'
WHERE time >= now() - interval '1 hours'
AND ('count' IS NOT NULL OR 'density' IS NOT NULL)"""

table = db.query(query=query, language='sql')
df = table.to_pandas().sort_values(by="time", ascending=False)			# pd.DataFrame 변환		/ to_pylist() : 리스트로 변환
print(df.columns) # max, min, time

data = {'count':df.iloc[:, 0], 'density':df.iloc[:,1], 'id':df.iloc[:,2], 'time':df.iloc[:,3]}
print(data)

# data = [{"timestamp": row[1], "population": row[2]} for row in df]