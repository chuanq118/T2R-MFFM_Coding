import os.path

import pandas as pd

if __name__ == '__main__':
    raw_csv_path = os.path.join(r'taxi_february_processed.csv')
    raw_data: pd.DataFrame = pd.read_csv(raw_csv_path)
    # raw_data.columns = ['id', 'time', 'coord']
    # lat: list = []
    # long: list = []
    # for c in raw_data['coord']:
    #     lat.append(str(c).split(' ')[1][:-1])
    #     long.append(str(c).split(' ')[0][6:])
    # raw_data['latitude'] = lat
    # raw_data['longitude'] = long
    # raw_data = raw_data.drop(columns=['coord'], axis=1)
    raw_data.to_csv('taxi_february_processed.csv')
