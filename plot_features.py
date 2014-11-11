#!/usr/bin/env python
#
# Display scatter plots of various features with different symbols
# for preictal and interictal segments.

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import load_data
import features

data_dir = os.path.abspath('data/Dog_1')
data_files = os.listdir(data_dir)

# set up plot
fig = plt.figure(figsize=(8,8))
fig.set_tight_layout(True)

# define functions and labels for features
feature_functions = [features.mean_of_means,
                     features.std_of_means,
                     features.mean_of_stds,
                     features.std_of_stds,
                     features.max_dev]
feature_labels = ['mean',
                  'std. dev. of means',
                  'mean of std. devs.',
                  'std. dev. of std. devs.',
                  'max dev.']

# compute and plot features for each segment
for seg_type, color, marker in zip(['interictal', 'preictal'],
                                   ['k', 'r'],
                                   ['.', 'x']):
    X = np.zeros(len(feature_functions))
    for f in data_files:
        if f.split('.')[-1] == 'mat' and seg_type in f:
            data = load_data.load_data(os.path.join(data_dir, f))
            new_features, columns = features.compute_features(data['data'],
                                                              feature_functions,
                                                              feature_labels)
            X = np.vstack((X, new_features))
    X = X[1:,:]

    n_features = len(feature_functions)
    for i in range(1,n_features):
        for j in range(i):
            plt.subplot(n_features-1, n_features-1, (n_features-1)*(i-1)+j+1)
            plt.scatter(X[:,j], X[:,i], c=color, marker=marker, s=20)
            if i == len(feature_functions)-1:
                plt.xlabel(feature_labels[j])
            if j == 0:
                plt.ylabel(feature_labels[i])

plt.show()
    
