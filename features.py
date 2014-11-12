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

def binned_power_spectrum(data, freq_bin_edges, sampling_rate):
    freq = np.fft.rfftfreq(data.size, d=1./sampling_rate)
    power = np.abs(np.fft.rfft(data))**2
    return np.histogram(freq, freq_bin_edges, weights=power)[0] / \
           np.histogram(freq, freq_bin_edges)[0]
                       
def power_mean(data, sampling_rate_hz):
    # define frequency bins (units = Hz)
    freq_bin_edges = np.logspace(-2, 2, num=6)

    # compute binned power spectrum for each electrode
    power = np.zeros(len(freq_bin_edges)-1)
    for i in range(data.shape[0]):
        power_i = binned_power_spectrum(data[i,:], freq_bin_edges,
                                        sampling_rate_hz)
        power = np.vstack((power, power_i))
    power = power[1:,:]

    # compute the mean in each frequency bin
    return np.mean(power, axis=0)

def power_cov(data, sampling_rate_hz):
    # define frequency bins (units = Hz)
    freq_bin_edges = np.logspace(-2, 2, num=6)

    # compute binned power spectrum for each electrode
    power = np.zeros(len(freq_bin_edges)-1)
    for i in range(data.shape[0]):
        power_i = binned_power_spectrum(data[i,:], freq_bin_edges,
                                        sampling_rate_hz)
        power = np.vstack((power, power_i))
    power = power[1:,:]

    # compute the covariance between frequency bins
    cov = np.cov(power, rowvar=0)
    # order of returned elements is 11, 12, ..., 1N, 22, 23, ..., NN
    return cov[np.triu_indices(len(cov))]


def compute_features(data, functions, function_args):
    """
    Given a matrix with EEG voltage time series in rows
    (one row for each electrode recorded in a particular segment),
    and a dictionary of functions to apply to each time series, return
    an array with the resulting features and a list of column labels.
    """
    features = []
    for f, args in zip(functions, function_args):
        if args is None:
            new_features = f(data)
        else:
            new_features = f(data, *args)
        try:
            features.extend(new_features)
        except:
            features.append(new_features)
    return features

def compute_feature_matrix(data_dir, functions, function_args, labels,
                           save_file=None):
    """
    For each .mat EEG data file in data_dir, compute the features given
    by functions and labels and return a 2D array where each row contains
    the index of the hour the segment belongs to, the segment type
    ('preictal': 1, 'interictal': 0, 'test': -1), and its features.
    Save the resulting feature matrix if the save_file keyword is set.
    Additional arguments required to evaluate function should be passed
    in function_args (use None for functions that need no extra arguments).
    """
    X = np.zeros(len(labels) + 2) # add 2 columns for hour and type
    data_files = []
    
    for f in os.listdir(data_dir):
        if f.split('.')[-1] == 'mat':
            data_files.append(f)
            data = load_data.load_data(os.path.join(data_dir, f))
            new_features = compute_features(data['data'], functions,
                                            function_args)
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
