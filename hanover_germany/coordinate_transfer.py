import pandas as pd
import pyproj

df = pd.read_csv(r'C:\Users\legen\Documents\001-论文\数据集\Hannover, Germany\hannover_table.csv')


# 创建UTM投影对象
utm_zone_number = 32  # 选择所在UTM带的带号
utm_northern = True   # True表示北半球，False表示南半球
utm_proj = pyproj.Proj(proj='utm', zone=utm_zone_number, north=utm_northern)


def utm_to_latlong(north_utm, east_utm):
    utm_proj = pyproj.Proj(proj='utm', zone=utm_zone_number, north=utm_northern, ellps='WGS84')
    wgs84_proj = pyproj.Proj(proj='latlong', zone=utm_zone_number, north=utm_northern, datum='WGS84')
    longitude, latitude = pyproj.transform(utm_proj, wgs84_proj, east_utm, north_utm)
    return latitude, longitude


def handle_row(row):
    coordinate = utm_to_latlong(row['north_utm'], row['east_utm'])
    row['latitude'] = coordinate[0]
    row['longitude'] = coordinate[1]


df.apply(handle_row, axis=1)
# 转换UTM坐标为经纬度坐标
# df['longitude'] = df.apply(lambda row: utm_proj(row['east_utm'], row['north_utm'], inverse=True)[0], axis=1)
# df['latitude'] = df.apply(lambda row: utm_proj(row['east_utm'], row['north_utm'], inverse=True)[1], axis=1)

# 保存为新文件
output_file = r'C:\Users\legen\Documents\001-论文\数据集\Hannover, Germany\hannover_table_with_coords_2.csv'
df.to_csv(output_file, index=False)

print(f"转换完成并保存为: {output_file}")
