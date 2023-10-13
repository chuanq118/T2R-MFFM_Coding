import os
import pandas as pd

data_dir = 'split'

files = os.listdir(data_dir)

min_latitude = None
max_latitude = None

min_longitude = None
max_longitude = None

# 变量所有的数据文件,获取最大/小的经纬度
for f in files:
    df = pd.read_csv(os.path.join(data_dir, f), header=None, names=['id', 'time', 'longitude', 'latitude'])

    min_lat = df['latitude'].min()
    max_lat = df['latitude'].max()
    min_long = df['longitude'].min()
    max_long = df['longitude'].max()

    if min_latitude is None or min_lat < min_latitude:
        min_latitude = min_lat
    if max_latitude is None or max_lat > max_latitude:
        max_latitude = max_lat

    if min_longitude is None or min_long < min_longitude:
        min_longitude = min_long
    if max_longitude is None or max_long > max_longitude:
        max_longitude = max_long

with open('region_coordinate.txt', 'w', encoding='utf-8') as f:
    f.write(f'min_latitude={min_latitude}')
    f.write(f'max_latitude={max_latitude}')
    f.write(f'min_longitude={min_longitude}')
    f.write(f'max_longitude={max_longitude}')
