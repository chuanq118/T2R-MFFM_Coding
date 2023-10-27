import pandas as pd
import numpy as np
import cv2

params = {
    'resolution': 512,
}

# 创建空白图像
traj_line_img = np.zeros((params['resolution'], params['resolution']), dtype=np.int8)
# 读取数据
df = pd.read_csv(r'data/region_demo_(512, 512)_indexed.csv', header=0)
# 按轨迹编号分组
trips = df.groupby('trip_id')

for _, trip_df in trips:
    # 按时间排序
    trip_df = trip_df.sort_values(by='unixtime')
    # 画线
    pre = trip_df.iloc[0]
    for i in range(1, len(trip_df)):
        cur = trip_df.iloc[i]
        # noinspection PyTypeChecker
        cv2.line(traj_line_img,
                 (int(pre['tile_lon_idx']), int(pre['tile_lat_idx'])),
                 (int(cur['tile_lon_idx']), int(cur['tile_lat_idx'])),
                 255,  # BGR
                 1,
                 lineType=cv2.LINE_AA)  # cv2.LINE_AA,抗锯齿线型
        pre = cur

cv2.imwrite(f'data/region_demo_{params["resolution"]}_trajectory_line.png', traj_line_img)
