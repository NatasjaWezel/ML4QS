"""
Question 4 of chapter 3:
Similarly to what we have done for our crowdsignals dataset,apply the techniques
that have been discussed in this chapter to the dataset you have collected yourself.
Write down your observations and argue for certain choices you have made.
"""

from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters


"""
Load the granularized dataset
"""

DATASET_DIR = Path('./intermediate_datafiles/chapt2/')
SMALL_GRAN_FILENAME = '250_ms_gran.csv'
LARGE_GRAN_FILENAME = '1000_ms_gran.csv'
dataset_small = pd.read_csv(
    Path(DATASET_DIR / SMALL_GRAN_FILENAME), index_col=0)
dataset_small.index = pd.to_datetime(dataset_small.index)

dataset_large = pd.read_csv(
    Path(DATASET_DIR / LARGE_GRAN_FILENAME), index_col=0)
dataset_large.index = pd.to_datetime(dataset_large.index)

DATASETS = {'250_ms_gran': dataset_small, '1000_ms_gran': dataset_large}
DATA_COLUMNS = (
    'acceleration_X', 'acceleration_Y', 'acceleration_Z', 'Illuminance',
    'accelerationlinear_X', 'accelerationlinear_Y', 'accelerationlinear_Z',
    'compass_X', 'compass_Y', 'compass_Z', 'Latitude', 'Longitude',
    'Altitude', 'gravity_X', 'gravity_Y', 'gravity_Z', 'gyro_X',
    'gyro_Y', 'gyro_Z', 'rotation_X', 'rotation_Y', 'rotation_Z',
    'rotation_cos', 'rotation_headingAccuracy'
)

viz = VisualizeDataset(__file__)

"""
Lets first analyse the distribution of the measured variables

for ds in DATASETS:
    for col in DATA_COLUMNS:
        print(DATASETS[ds][col].shape)

        plt.hist(DATASETS[ds][col], bins=40)

        plt.title(f'Distribution of data {col} for gran {ds}')

        plt.show()
"""
"""
Lets choose the measurement acceleration_X
and process the 250ms as well as the 1000 ms aggregation.
First we want to remove outliers.
"""
col = 'acceleration_X'

"""
Lets see how the col is distributed roughly
"""
for ds in DATASETS:
    # plt.hist(DATASETS[ds][col], bins=40)
    # plt.title(f'Distribution of data {col} for gran {ds}')
    # plt.show()
    pass


"""
Lets analyse the outliers under the assumption the set is normal distributed
"""

outlier_detection = DistributionBasedOutlierDetection(mixture_comp=2)
for ds in DATASETS:
    dataset = DATASETS[ds]
    dataset = outlier_detection.chauvenet(dataset, col)
    # viz.plot_binary_outliers(dataset, col, col + '_outlier')

"""
acceleration_X might be a mixture model of two gaussian distributions
Lets look at the likelihood of the observed measurements under this assumption:
"""

for ds in DATASETS:
    dataset = DATASETS[ds]
    dataset = outlier_detection.mixture_model(dataset, col)
    # viz.plot_dataset(
    #    dataset, [col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'])

"""
Lets apply chauv criterion to all cols for the two granularities
and save them as interm result
"""
for ds in DATASETS:
    dataset = DATASETS[ds]
    for c in dataset.columns:
        if '_outlier' in c or '_mixture' in c or 'lof' in c or 'simple_dist_outlier' in c:
            if c in dataset.columns:
                dataset.pop(c)
            continue
        dataset = outlier_detection.chauvenet(dataset, c)
        dataset.loc[dataset[f'{c}_outlier'] == True, c] = np.nan
        del dataset[c + '_outlier']
    dataset.to_csv(DATASET_DIR / (ds+'_outliers.csv'))


"""
Now impute missing values
"""
# change dataset to be the outlier data set
dataset_small = pd.read_csv(
    Path(DATASET_DIR / '250_ms_gran_outliers.csv'), index_col=0)
dataset_small.index = pd.to_datetime(dataset_small.index)

dataset_large = pd.read_csv(
    Path(DATASET_DIR / '1000_ms_gran_outliers.csv'), index_col=0)
dataset_large.index = pd.to_datetime(dataset_large.index)


for ds in DATASETS:
    dataset = DATASETS[ds]
    for c in dataset.columns:
        MisVal = ImputationMissingValues()
        imputed_mean_dataset = MisVal.impute_mean(
            deepcopy(dataset), c)
        imputed_median_dataset = MisVal.impute_median(
            deepcopy(dataset), c)
        imputed_interpolation_dataset = MisVal.impute_interpolate(
            deepcopy(dataset), c)
        viz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], c,
                                imputed_mean_dataset[c], imputed_interpolation_dataset[c])
