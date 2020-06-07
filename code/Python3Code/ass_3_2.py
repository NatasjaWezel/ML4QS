"""
In the example the following values are used to find the ouliers
• Chauvenet’s criterion: we set the value c = 2, according to the traditional Chau- venet criterion.
• Mixture models: we use 3 mixture components and a single iteration of the algo- rithm
• Simple distance-based approach: we set dmin = 0.1 and fmin = 0.99 and use Euclidean distance.
• Local outlier factor: we use 5 for the value of k and Euclidean distance.

Question:
Vary the constant c (smaller and larger values) of the Chauvenet’s criterion and 
study the dependency of the number of detected outliers on c.
Repeat this for the other three methods presented for outlier detection.
Use the source code from book’s website, that generated the figures, as a starting point for the analysis.


Context:
Chauvenets Criterion states that a measurement is labeled as an outlier if the prob of observing the value
is lower than 1/(cN).

Other three methods are:
* Gauss mixed model
* Simple Distance-Based Approach
* Local Outlier Factor
"""

import sys
import copy

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection

DataViz = VisualizeDataset(__file__)

DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter2_result.csv'

dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
dataset.index = pd.to_datetime(dataset.index)

outlier_columns = ['acc_phone_x', 'light_phone_lux']


def chauvenet_outliers(c, col, dataset):
    outlier_dist = DistributionBasedOutlierDetection(threshold=c)
    dataset = outlier_dist.chauvenet(dataset, col)
    #DataViz.plot_binary_outliers(dataset, col, col + '_outlier', title=f"Run for c is {c}")
    return len([outlier for outlier in dataset[col + '_outlier'] if outlier])


def mixture(comps, col, dataset, c=2):
    """
    Takes number of component
    """
    outlier_dist = DistributionBasedOutlierDetection(
        mixture_comp=comps, threshold=c)
    dataset = outlier_dist.mixture_chauvenet(dataset, col)
    return len([outlier for outlier in dataset[col + '_outlier'] if outlier])


def simple_distance(dataset, col, dmin, fmin):
    dataset = DistanceBasedOutlierDetection().simple_distance_based(
        dataset, [col], 'euclidean', dmin, fmin)
    return len([outlier for outlier in dataset['simple_dist_outlier'] if outlier])


def local_oulier(dataset, col, d):

    ds = DistanceBasedOutlierDetection().local_outlier_factor(
        dataset, [col], 'euclidean', d)
    return len([outlier for outlier in ds['lof'] if outlier])


for outlier_column in outlier_columns:

    print(dataset[outlier_column])
    plt.hist(dataset[outlier_column], bins=200)
    plt.title(f'Histogram of measurements for col {outlier_column}')
    plt.show()
'''
for col in outlier_columns:
    outliers = np.zeros(30)
    c_values = np.linspace(.05, 4, num=30)
    for i, c in enumerate(c_values):
        number_outliers = chauvenet_outliers(c, col, dataset)
        outliers[i] = number_outliers
    plt.plot(c_values, outliers)
    plt.xlabel('c')
    plt.ylabel('Number outliers')
    plt.title(f'Number of outliers dependent on c for column {col}')
    plt.show()

for col in outlier_columns:
    if col == 'acc_phone_x':
        k = 3
    outliers = np.zeros(78)
    k_values = range(2, 80)  # np.linspace(.05, 4, num=30)
    for i, k in enumerate(k_values):
        number_outliers = mixture(k, col, dataset, c=6)
        outliers[i] = number_outliers
    plt.plot(k_values, outliers)
    plt.xlabel('K')
    plt.ylabel('Number outliers')
    plt.title(
        f'Number of outliers dependent on K for column {col} using mixture model c=2')
    plt.show()
for col in outlier_columns:
    outliers = np.zeros((10, 10))
    dmin = np.linspace(.8, 2., num=10)
    fmin = np.linspace(.90, 1.1, num=10)
    for i, d in enumerate(dmin):
        for j, f in enumerate(fmin):
            outliers[i][j] = simple_distance(dataset, col, d, f)
            # outliers[i][j] = number_outliers
    # ([dmin,], outliers) 0.10, 0.99
    plt.imshow(outliers, extend=[.8, 2.0, 0.9, 1.1])
    plt.xlabel('dmin')
    plt.ylabel('fmin')
    plt.title(f'Number of outliers dependent on c for column {col}')
    plt.show()


for col in outlier_columns:
    outliers = np.zeros((10, 10))
    ds = range(3, 10)  # np.linspace(4, 6, num=10)
    for i, k in enumerate(ds):
        outliers[i] = local_oulier(dataset, col, k)

    plt.plot(ds, outliers)
    plt.xlabel('k')
    plt.ylabel('Number outliers')
    plt.title(
        f'Number of outliers dependent on c for column {col} using mixture model k={k}')
    plt.show()
'''
'''
outliers = np.zeros(5)
ds = (2, 4, 6, 8, 14) #range(3, 10)  # np.linspace(4, 6, num=10)
for i, k in enumerate(ds):
    outliers[i] = local_oulier(dataset, col, k)
plt.plot(ds, outliers)
plt.xlabel('k')
plt.ylabel('Number outliers')
plt.title(
    f'Number of outliers dependent on c for column {col} using mixture model k={k}')
plt.show()
'''
