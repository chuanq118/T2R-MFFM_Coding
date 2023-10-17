import pyproj

# 创建UTM投影对象
utm_zone_number = 32  # 选择所在UTM带的带号
utm_northern = True   # True表示北半球，False表示南半球
utm_proj = pyproj.Proj(proj='utm', zone=utm_zone_number, north=utm_northern)

# 输入UTM坐标
utm_easting = 550800.9916  # 举例中的东方坐标
utm_northing = 5809940.175  # 举例中的北方坐标

# 转换为经纬度坐标
lon, lat = utm_proj(utm_easting, utm_northing, inverse=True)

print(f"经度 (Longitude): {lon}")
print(f"纬度 (Latitude): {lat}")


