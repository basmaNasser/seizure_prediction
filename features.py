# Define functions to compute features from the EEG time series data.

import os
import numpy as np
import load_data

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

def compute_features(data, functions):
    """
    Given a matrix with EEG voltage time series in rows
    (one row for each electrode recorded in a particular segment),
    and a dictionary of functions to apply to each time series, return
    an array with the resulting features and a list of column labels.
    """
    features = []
    for f in functions:
        new_features = f(data)
        try:
            features.extend(new_features)
        except:
            features.append(new_features)
    return features

def compute_feature_matrix(data_dir, functions, labels, save_file=None):
    """
    For each .mat EEG data file in data_dir, compute the features given
    by functions and labels and return a 2D array where each row contains
    the index of the hour the segment belongs to, the segment type
    ('preictal': 1, 'interictal': 0, 'test': -1), and its features.
    Save the resulting feature matrix if the save_file keyword is set.
    """
    X = np.zeros(len(labels) + 2) # add 2 columns for hour and type
    data_files = []
    
    for f in os.listdir(data_dir):
        if f.split('.')[-1] == 'mat':
            data_files.append(f)
            data = load_data.load_data(os.path.join(data_dir, f))
            new_features = compute_features(data['data'], functions)
            if data['type'] == 'preictal':
                seg_type = 1
            elif data['type'] == 'interictal':
                seg_type = 0
            elif data['type'] == 'test':
                seg_type = -1
            else:
                seg_type = np.nan
            new_features = np.hstack(([data['hour'], seg_type],
                                      new_features))
            X = np.vstack((X, new_features))

    X = X[1:,:]

    if save_file is not None:
        columns = ['hour', 'type'] + labels
        np.savetxt(save_file, X, fmt='%.4e',
                   header='Data directory: ' + data_dir + \
                        '\nColumns:\n ' + '\n '.join(columns))
        data_list_file = '.'.join(save_file.split('.')[:-1]) + '_data_files.txt'
        with open(data_list_file, 'w') as df:
            df.writelines('\n'.join(data_files))
        
    return (X, data_files)

def scale_features(feature_matrix, exclude_columns=[0, 1]):
    """
    Given a feature matrix with each row containing a feature vector
    for a particular instance, rescale each feature to have mean zero
    and unit variance. Specify columns to exclude from rescaling with
    exclude_columns (1st 2 columns are excluded by default since they
    are assumed to list the hour index and segment type).
    """
    X = np.copy(feature_matrix)
    for i in range(X.shape[1]):
        if i not in exclude_columns:
            X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
    return X
