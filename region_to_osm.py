import os.path

from lxml import etree
import pandas as pd

# 以 OSM 的格式生成该区域内包含所有轨迹点的 XML 文件
trajectories_file_path = "trajectory/1.trajectories.csv"
osm_file_path = 'osm/1.osm'

base_filename = os.path.basename(osm_file_path).split(".")[0]

# 解析获取 bounds 区域数据
osm_bounds = etree.parse(osm_file_path).getroot().find('bounds')

# 设置根节点
osm = etree.Element('osm')
doc = etree.ElementTree(osm)

# 根节点设置属性
osm.set("version", "1.0")

# 写入 bounds 节点
bounds = etree.SubElement(osm, "bounds")
bounds.set('minlat', osm_bounds.get('minlat'))
bounds.set('maxlat', osm_bounds.get('maxlat'))
bounds.set('minlon', osm_bounds.get('minlon'))
bounds.set('maxlon', osm_bounds.get('maxlon'))

df = pd.read_csv(trajectories_file_path, header=None, names=['id', 'time', 'longitude', 'latitude'])

trajectory_id_counter = 10000
way_id_counter = 100000000

way_map = {}

# 遍历每个轨迹点 生成 node 节点
for idx, row in df.iterrows():
    trajectory_id_counter += 1
    car_id = str(int(row['id']))
    nid = str(trajectory_id_counter)
    # 如果不存在此轨迹线,则初始化一个
    if car_id not in way_map:
        way_id_counter += 1
        way_map[car_id] = etree.SubElement(osm, "way")
        way_map[car_id].set('id', str(way_id_counter))
        way_map[car_id].set('visible', 'true')
        way_map[car_id].set('cid', car_id)
    # 添加一个轨迹点
    node = etree.SubElement(osm, 'node')
    node.set('id', nid)
    node.set('visible', 'true')
    node.set('timestamp', str(row['time']))
    node.set('cid', car_id)
    node.set('lat', str(row['latitude']))
    node.set('lon', str(row['longitude']))
    # 将轨迹点添加入轨迹线中
    nd = etree.SubElement(way_map[car_id], "nd")
    nd.set('ref', nid)

with open(f'osm/{base_filename}.trajectories.osm', 'wb') as f:
    # 设置打印 XML 声明头 / 美化打印
    doc.write(f, xml_declaration=True, pretty_print=True, encoding='utf-8')
