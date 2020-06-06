from pathlib import Path
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt

from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset

DATA_BASE_DIR = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter3_result_outliers.csv'

dataset = pd.read_csv(Path(DATA_BASE_DIR / DATASET_FNAME), index_col=0)
dataset.index = pd.to_datetime(dataset.index)
viz = VisualizeDataset(__file__)


# first linearly interpolate
karlman_dataset = ImputationMissingValues().impute_interpolate(
    deepcopy(dataset), 'hr_watch_rate')
karlman_dataset = KalmanFilters().apply_kalman_filter(
    karlman_dataset, 'hr_watch_rate')
viz.plot_imputed_values(dataset, [
    'original', 'Karlman filter'], 'hr_watch_rate', karlman_dataset['hr_watch_rate_kalman'])
