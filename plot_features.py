#!/usr/bin/env python
#
# Display scatter plots of various features with different symbols
# for preictal and interictal segments.

import os.path
import numpy as np
import matplotlib.pyplot as plt
import features

data_dir = os.path.abspath('data/Dog_1')
feature_file = os.path.join(data_dir, 'features_01.txt')

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

# compute and save features, or laod previously computed features
if os.path.isfile(feature_file):
    X = np.loadtxt(feature_file)
else:
    X, data_files = features.compute_feature_matrix(data_dir,
                                            feature_functions, feature_labels,
                                            save_file=feature_file)

X = features.scale_features(X)

# plot features for each segment
for seg_type, color, marker in zip([0, 1],
                                   ['k', 'r'],
                                   ['.', 'x']):
    # select rows matching a given segment type and remove (hour, type) cols.
    seg_features = X[X[:,1] == seg_type, 2:]

    n_features = len(feature_functions)
    for i in range(1,n_features):
        for j in range(i):
            plt.subplot(n_features-1, n_features-1, (n_features-1)*(i-1)+j+1)
            plt.scatter(seg_features[:,j], seg_features[:,i],
                        c=color, marker=marker, s=20)
            if i == len(feature_functions)-1:
                plt.xlabel(feature_labels[j])
            if j == 0:
                plt.ylabel(feature_labels[i])

plt.show()
    
