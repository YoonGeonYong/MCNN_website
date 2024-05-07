import os, time
from influxdb_client_3 import InfluxDBClient3, Point
import numpy as np

host = 'https://us-east-1-1.aws.cloud2.influxdata.com'
token = os.environ.get("INFLUXDB_TOKEN")
org = 'Dev team'

client = InfluxDBClient3(host=host, token=token, org=org)
database="density"

# data insert
for i in range(10):
	_max = np.random.randint(0,100)
	_min = np.random.randint(0,100)

	point = (
		Point('test') # measurement
		.tag('max', _max)
		.field('min', _min)
	)
	client.write(database=database, record=point)

print("Complete. Return to the InfluxDB UI.")


# data select
query = """SELECT *
FROM 'test'
WHERE time >= now() - interval '1 hours'
AND ('bees' IS NOT NULL OR 'ants' IS NOT NULL)"""

# Execute the query
table = client.query(query=query, database="density", language='sql')

# Convert to dataframe
df = table.to_pandas().sort_values(by="time")
print(df)