# Define functions to compute features from the EEG time series data.

import numpy as np


def mean_of_means(data):
    return np.mean(data)

def std_of_means(data):
    return np.std(np.mean(data, axis=1))

def mean_of_stds(data):
    return np.mean(np.std(data, axis=1))

def std_of_stds(data):
    return np.std(np.std(data, axis=1))

def max_dev(data):
    return np.max(np.abs(data.T - np.mean(data, axis=1)))

def compute_features(data, functions, labels):
    """
    Given a matrix with EEG voltage time series in rows
    (one row for each electrode recorded in a particular segment),
    and a dictionary of functions to apply to each time series, return
    an array with the resulting features and a list of column labels.
    """
    features = []
    columns = []
    for f, col in zip(functions, labels):
        features.append(f(data))
        columns.append(col)
    return (features, columns)
