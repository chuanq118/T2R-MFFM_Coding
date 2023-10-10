import os.path

import pandas as pd
import logging
import sys

logging.basicConfig(level=logging.INFO)

if len(sys.argv) <= 1:
    logging.error('需要指定数据集文件路径')
    sys.exit(1)

df = pd.read_csv(sys.argv[1], sep=';', header=None)

total_len = len(df)
logging.info('读取数据完成,总行数为{}'.format(total_len))

df.columns = ['id', 'time', 'coord']

logging.info('设置列名id, time, coord')

# df['time'] = pd.to_datetime(df['time'], format='mixed')

err_indices: list = []

# df['timestamp'] = df['time'].apply(lambda x: x.timestamp())
#
# logging.info('转换时间列为时间戳格式')
#
# df['latitude'] = df['coord'].apply(lambda x: x.split()[1])
# df['longitude'] = df['coord'].apply(lambda x: x.split()[0][6:])
#
# logging.info('解析坐标为纬度经度')
#
# df = df.drop(columns=['coord'])

# batch_size = 1000
# print(df)
# df2=df.copy(deep=True)
# df2 = df2.drop(columns=['coord'], axis=1)
# print(df2)

if not os.path.exists('split'):
    os.mkdir('split')

opened_files: dict = {
    'err': open(f'split/err.txt', 'w', encoding='utf-8')
}

count = 0

for index, row in df.iterrows():
    count += 1
    if count % 10000 == 0:
        print(f'-> 处理已完成 {count}/{total_len}')
    try:
        _time = int((pd.to_datetime(row['time'], format='mixed').timestamp() * 1000))
        _latitude = row['coord'].split()[1][:-1]
        _longitude = row['coord'].split()[0][6:]
        _id = row['id']
        if _id not in opened_files:
            opened_files[_id] = open(f'split/{_id}.txt', 'w', encoding='utf-8')
        opened_files[_id].write(','.join([str(_id), str(_time), str(_latitude), str(_longitude)]) + '\n')
        opened_files[_id].flush()
    except Exception as e:
        logging.error(e)
        opened_files['err'].write('{};{};{}\n'.format(row['id'], row['time'], row['coord']))
        opened_files['err'].flush()


# for i in range(0, len(df), batch_size):
#     batch_df = df.loc[i:i + batch_size - 1].copy()
#
#     logging.info('处理批次{}/{}行'.format(i, i + batch_size))
#
#     for id_ in batch_df['id'].unique():
#         id_df = batch_df[batch_df['id'] == id_]
#         id_df.to_csv(f'split/{id_}.txt', sep=';', index=False, header=False)
#
#     logging.info('================ 批次处理完毕 ====================')
#     break
#
# logging.info('全部处理完成')
