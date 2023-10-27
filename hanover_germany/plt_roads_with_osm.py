"""
依据 OSM 数据
绘制区域对应的道路
"""
import math

from lxml import etree
import torch
import matplotlib.pyplot as plt
import cv2

# osm 源数据文件(已提取道路的)
osm_source = r'osm/hanover_512x512_map-roads.osm'
# 指定分辨率 512 x 512
resolution = 512

vector = torch.zeros(resolution, resolution)

# 解析 xml 文档
doc = etree.parse(osm_source)
osm = doc.getroot()
# 读取区域
bounds = osm.find('bounds')
min_longitude = float(bounds.get('minlon'))
min_latitude = float(bounds.get('minlat'))
max_longitude = float(bounds.get('maxlon'))
max_latitude = float(bounds.get('maxlat'))

# 计算 tile 间隔
latitude_interval = (max_latitude - min_latitude) / resolution
longitude_interval = (max_longitude - min_longitude) / resolution
print(f'latitude_interval = {latitude_interval}')
print(f'longitude_interval = {latitude_interval}')

# 缓存所有 node 节点
node_map = {}
for node in osm.findall(".//node"):
    node_map[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))


def bresenham_line(x1, y1, x2, y2):
    """
    Bresenham 算法利用了直线的斜率来决定每一步应该向哪个方向移动，并选择距离直线更接近的像素点来填充线段
    :param x1: 起始点行坐标
    :param y1: 起始点列坐标
    :param x2: 结束点行坐标
    :param y2: 结束点列坐标
    :return: 连接起始点和结束点,连线上元素值设置为 1
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            # 一定要注意边界判断,否则会出现意料之外的线段!!!
            if x >= resolution or y >= resolution or x < 0 or y < 0:
                break
            vector[x, y] = 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            # 一定要注意边界判断,否则会出现意料之外的线段!!!
            if x >= resolution or y >= resolution or x < 0 or y < 0:
                break
            vector[x, y] = 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy


def plt_line(start_point, end_point):
    """
    绘制两个点之间的路线
    :param start_point: (lat, lon)
    :param end_point:  (lat, lon)
    :return:
    """
    start_index_row: int = math.floor((max_latitude - start_point[0]) / latitude_interval)
    start_index_col: int = math.floor((start_point[1] - min_longitude) / longitude_interval)
    end_index_row: int = math.floor((max_latitude - end_point[0]) / latitude_interval)
    end_index_col: int = math.floor((end_point[1] - min_longitude) / longitude_interval)
    # 确保起始点均在区域内部(会忽略掉起始点在区域外但结束点在区域内的路段)
    # > -1 > -1
    if (resolution > start_index_row > -1) and (resolution > start_index_col > -1):
        bresenham_line(start_index_row, start_index_col, end_index_row, end_index_col)
    else:
        # 如果结束点仍然在区域内,接着绘制
        if -1 < end_index_row < resolution and -1 < end_index_col < resolution:
            max_col_ = end_index_col if end_index_col > start_index_col else start_index_col
            min_col_ = end_index_col if end_index_col < start_index_col else start_index_col
            max_row_ = end_index_row if end_index_row > start_index_row else start_index_row
            min_row_ = end_index_row if end_index_row < start_index_row else start_index_row

            _k = (end_index_row - start_index_row) / (end_index_col - start_index_col)
            _b = end_index_row - _k * end_index_col
            # 测试与 row = resolution 相交的点
            y_ = resolution - 1
            x_ = int((y_ - _b) / _k)
            if x_ < 0 or x_ >= resolution or x_ < min_col_ or x_ > max_col_:
                # 改为 col = resolution 相交点
                x_ = resolution - 1
                y_ = int(_k * x_ + _b)
            if 0 <= y_ < resolution and max_row_ > y_ > min_row_:
                print(f'x_ = {x_}, y_ = {y_}')
                print(f'end_index_row = {end_index_row}, end_index_col = {end_index_col}')
                bresenham_line(y_, x_, end_index_row, end_index_col)


# 找到所有的路线并连线
for way in osm.findall(".//way"):
    # vector = torch.zeros(resolution, resolution)  # for test
    # way_id = way.get('id')  # for test
    nodes = way.findall("nd")
    for i in range(len(nodes) - 1):
        plt_line(node_map[nodes[i].get('ref')], node_map[nodes[i + 1].get('ref')])
    # break
    # vector = vector * 255  # for test
    # plt.imsave(f'data/region_demo_roads_{way_id}.png', vector, cmap='gray', vmin=0, vmax=255)  # for test

vector = vector * 255
# 保存图片
plt.imsave('data/region_demo_roads.png', vector, cmap='gray', vmin=0, vmax=255)
cv2.imwrite('data/region_demo_roads_cv2.png', vector.numpy())
