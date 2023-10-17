import pandas as pd
import pyproj

df = pd.read_csv(r'C:\Users\legen\Documents\1A-论文\数据集\Hannover, Germany\hannover_table.csv')


# 创建UTM投影对象
utm_zone_number = 32  # 选择所在UTM带的带号
utm_northern = True   # True表示北半球，False表示南半球
utm_proj = pyproj.Proj(proj='utm', zone=utm_zone_number, north=utm_northern)

# 转换UTM坐标为经纬度坐标
df['longitude'] = df.apply(lambda row: utm_proj(row['east_utm'], row['north_utm'], inverse=True)[0], axis=1)
df['latitude'] = df.apply(lambda row: utm_proj(row['east_utm'], row['north_utm'], inverse=True)[1], axis=1)

# 保存为新文件
output_file = r'C:\Users\legen\Documents\1A-论文\数据集\Hannover, Germany\hannover_table_with_coords.csv'
df.to_csv(output_file, index=False)

print(f"转换完成并保存为: {output_file}")
