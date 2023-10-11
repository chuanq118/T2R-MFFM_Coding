import os

from lxml import etree
import pandas as pd

osm_file_path = 'osm/1.osm'
data_dir = 'split'

root = etree.parse(osm_file_path).getroot()
bounds = root.find('bounds')

min_longitude = float(bounds.get('minlon'))
min_latitude = float(bounds.get('minlat'))
max_longitude = float(bounds.get('maxlon'))
max_latitude = float(bounds.get('maxlat'))

# 提取数据输出的文件
osm_file_name = os.path.basename(osm_file_path).split(".")[0]
region_trajectories = open(f'./trajectory/{osm_file_name}.trajectories.csv', 'a', encoding='utf-8')

files = os.listdir(data_dir)
for f in files:
    print(f'############# read data file {f} ##############')
    df = pd.read_csv(os.path.join(data_dir, f), header=None, names=['id', 'time', 'longitude', 'latitude'])
    # 过滤经度范围
    df = df[(df['longitude'] > min_longitude) & (df['longitude'] < max_longitude)]
    # 过滤纬度范围
    df = df[(df['latitude'] > min_latitude) & (df['latitude'] < max_latitude)]

    df.to_csv(path_or_buf=region_trajectories, header=False, lineterminator='\n', index=None)
    region_trajectories.flush()

region_trajectories.close()

