from pathlib import Path

import pandas as pd


from util.VisualizeDataset import VisualizeDataset

viz = VisualizeDataset(__file__)

DATA_PATH = Path('./intermediate_datafiles/')
RESULT_FNAME = 'chapter4_result.csv'


periodic_predictor_cols = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z',
                           'acc_watch_x', 'acc_watch_y', 'acc_watch_z', 'gyr_phone_x', 'gyr_phone_y',
                           'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y',
                           'gyr_watch_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z',
                           'mag_watch_x', 'mag_watch_y', 'mag_watch_z']

dataset = pd.read_csv(
    Path(DATA_PATH / RESULT_FNAME), index_col=0)
dataset.index = pd.to_datetime(dataset.index)
for c in periodic_predictor_cols:
    cols = [f'{c}_max_freq', f'{c}_freq_weighted', f'{c}_pse', 'label']
    viz.plot_dataset(dataset, cols, [
                     'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
'''
viz.plot_dataset(dataset, ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label'], [
    'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])
'''
