# Define functions to compute features from the EEG time series data.

import os
import numpy as np
import load_data

def mean_of_means(data):
    return np.mean(data['data'])

def std_of_means(data):
    return np.std(np.mean(data['data'], axis=1))

def mean_of_stds(data):
    return np.mean(np.std(data['data'], axis=1))

def std_of_stds(data):
    return np.std(np.std(data['data'], axis=1))

def max_dev(data):
    return np.max(np.abs(data['data'].T - np.mean(data['data'], axis=1)))

def binned_power_spectrum(time_series, freq_bin_edges, sampling_rate):
    freq = np.fft.rfftfreq(time_series.size, d=1./sampling_rate)
    power = np.abs(np.fft.rfft(time_series))**2
    return np.histogram(freq, freq_bin_edges, weights=power)[0] / \
           np.histogram(freq, freq_bin_edges)[0]
                       
def power_mean(data):
    # define frequency bins (units = Hz)
    freq_bin_edges = np.logspace(-2, 2, num=6)

    # compute binned power spectrum for each electrode
    power = np.zeros(len(freq_bin_edges)-1)
    for i in range(data['data'].shape[0]):
        power_i = binned_power_spectrum(data['data'][i,:], freq_bin_edges,
                                        data['sampling_rate_hz'])
        power = np.vstack((power, power_i))
    power = power[1:,:]

    # compute the mean in each frequency bin
    return np.mean(power, axis=0)

def power_cov(data):
    # define frequency bins (units = Hz)
    freq_bin_edges = np.logspace(-2, 2, num=6)

    # compute binned power spectrum for each electrode
    power = np.zeros(len(freq_bin_edges)-1)
    for i in range(data['data'].shape[0]):
        power_i = binned_power_spectrum(data['data'][i,:], freq_bin_edges,
                                        data['sampling_rate_hz'])
        power = np.vstack((power, power_i))
    power = power[1:,:]

    # compute the covariance between frequency bins
    cov = np.cov(power, rowvar=0)
    # order of returned elements is 11, 12, ..., 1N, 22, 23, ..., NN
    return cov[np.triu_indices(len(cov))]


def compute_features(data, functions):
    """
    Given a data dictionary where data['data'] is a matrix with
    EEG voltage time series in rows
    (one row for each electrode recorded in a particular segment),
    and a list of functions to apply to each time series, return
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

def compute_feature_matrix(data_dir, functions, labels,
                           save_file=None, verbose=False):
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
            if verbose:
                print f
            data = load_data.load_data(os.path.join(data_dir, f))
            new_features = compute_features(data, functions)
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

def load_features(files, hour_column=0, type_column=1):
    """
    Read and combine feature matrices from multiple files,
    assumed to have the same set of columns.
    Hour indices in files after the first are incremented so
    that the indices for preictal and interictal subsamples
    remain unique, with only 6 segments per hour index.
    """
    n_hr_col_pre_tot = 0
    n_hr_col_inter_tot = 0
    
    for i, f in enumerate(files):
        X_i = np.loadtxt(f)
        
        # count the number of hour indices for preictal and interictal segments
        seg_type = X_i[:,type_column]
        n_hr_col_pre = len(np.unique(X_i[seg_type == 1,hour_column]))
        n_hr_col_inter = len(np.unique(X_i[seg_type == 0,hour_column]))
        
        # increment hour indices
        X_i[seg_type == 1,hour_column] += n_hr_col_pre_tot
        X_i[seg_type == 0,hour_column] += n_hr_col_inter_tot
        
        # combine feature matrices
        if i == 0:
            X = np.copy(X_i)
        else:
            X = np.vstack((X, X_i))
        n_hr_col_pre_tot += n_hr_col_pre
        n_hr_col_inter_tot += n_hr_col_inter

    return X

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

def remove_outliers(feature_matrix, n_sigma=2, verbose=False):
    """
    Given a feature matrix with each row containing a feature vector
    for a particular instance, identify instances grouped by the hour
    index (listed in column 0) for each preictal or interictal segment
    (type 0 or 1, listed in column 1) where the mean value of any
    feature is more than n_sigma standard deviations away from the
    rest of the instances, unless that mean value falls within the extent
    of that feature in the test data set (type -1).
    Return the feature matrix with outlier instance groups removed
    and the row indices of the remaining instances (including test data).
    """
    # copy matrix to avoid unintended changes
    X = np.copy(feature_matrix)
    indices = range(X.shape[0])
    # make masks for preictal, interictal, and test instances
    mask_pre = X[:,1] == 1
    mask_inter = X[:,1] == 0
    mask_test = X[:,1] == -1
    # extract the test instances
    X_test = X[mask_test,:]
    # get the hour indices for preictal and interictal samples
    hrs_pre = np.unique(X[mask_pre,0])
    hrs_inter = np.unique(X[mask_inter,0])

    # loop over the data set until no more outliers are found
    outliers_remain = True
    count = 0
    while outliers_remain:
        n_outliers = 0
        count += 1
        # loop over hour indices and segment types
        for i_hr, seg_type in zip(np.concatenate((hrs_pre, hrs_inter), axis=1),
                                  np.concatenate((np.repeat(1, len(hrs_pre)),
                                                  np.repeat(0, len(hrs_inter))),
                                                 axis=1)):
            mask_trial = (X[:,1] == seg_type) & (X[:,0] == i_hr)
            trial_indices = np.array(range(len(mask_trial)))[mask_trial]
            trial_hr = X[mask_trial]
            other_hrs = X[((X[:,1] != seg_type) | \
                          (X[:,0] != i_hr)) & \
                          (X[:,1] != -1),:]
            # compute feature statistics (excluding hour and type columns)
            trial_mean = np.mean(trial_hr[:,2:], axis=0)
            other_mean = np.mean(other_hrs[:,2:], axis=0)
            other_std = np.std(other_hrs[:,2:], axis=0)
            # check for outlier
            mean_outlier = np.abs(trial_mean-other_mean) > n_sigma*other_std
            lt_test_min = trial_mean < X_test[:,2:].min(axis=0)
            gt_test_max = trial_mean > X_test[:,2:].max(axis=0)
            if np.any(mean_outlier & (lt_test_min | gt_test_max)):
                n_outliers += 1
                # remove rows from feature matrix and index array
                X = np.delete(X, trial_indices, axis=0)
                indices = np.delete(indices, trial_indices)

        if verbose:
            print 'Pass ' + str(count) + ': ' + str(n_outliers) + \
                    ' outlier groups removed.'
        if n_outliers == 0:
            outliers_remain = False

        return X, indices
