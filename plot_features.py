#!/usr/bin/env python
#
# Display scatter plots of various features with different symbols
# for preictal and interictal segments.

import os.path
import numpy as np
import matplotlib.pyplot as plt
import features

data_dir = os.path.abspath('data/Dog_3')
feature_file = os.path.join(data_dir, 'features_01.txt')

# define functions and labels for features
feature_functions = [features.mean_of_means,
                     features.std_of_means,
                     features.mean_of_stds,
                     features.std_of_stds,
                     features.max_dev,
                     features.power_mean,
                     features.power_cov]
feature_labels = ['MM', 'SDM', 'MSD', 'SDSD', 'MaxD'] + \
                 ['PM' + str(i) for i in range(1, 6)] + \
                 ['PC' + str(i) + str(j) for i in range(1,6) \
                                         for j in range(i,6)]

# choose features to include in the plots
i_feature = np.array(range(len(feature_labels)))
#
#use_features = (i_feature < 5)
use_features = (i_feature > 4) & (i_feature < 10)
#use_features = (i_feature > 9)
#
feature_labels_used = list(np.array(feature_labels)[use_features])
n_features = len(feature_labels_used)

# fraction of interictal outliers to exclude in plot ranges
# - set to None to show all points
plot_outlier_fraction = None
# extra fraction of plot to show (not used if plot_outlier_fraction is None)
plot_f_edge = 0.05

# compute and save features, or laod previously computed features
if os.path.isfile(feature_file):
    X = np.loadtxt(feature_file)
    data_files = None
else:
    print 'Computing features...'
    X, data_files = features.compute_feature_matrix(data_dir,
                                            feature_functions, feature_labels,
                                            save_file=feature_file,
                                            verbose=True)

X, outlier_indices = features.remove_outliers(X, n_sigma=2, verbose=True)
if data_files is not None:
    data_files = np.delete(data_files, outlier_indices)

#X = features.scale_features(X)

# plot features for each segment
fig, axs = plt.subplots(n_features-1, n_features-1, figsize=(9,9),
                        sharex='col', sharey='row')
plt.subplots_adjust(hspace=0.001, wspace=0.001)
for i in range(n_features-1):
    for j in range(i+1, n_features-1):
        fig.delaxes(axs[i,j])
        
for seg_type, color, marker in zip([-1, 0, 1],
                                   ['b', 'y', 'r'],
                                   ['o', 'x', '+']):
    # select rows matching a given segment type and remove (hour, type) cols.
    seg_features = X[X[:,1] == seg_type, 2:]

    # select features to show in plot
    seg_features_used = seg_features[:,use_features]

    for i in range(1, n_features):
        for j in range(i):
            ax = axs[i-1, j]
            fx = seg_features_used[:,j]
            fy = seg_features_used[:,i]
            ax.scatter(fx, fy, edgecolors=color, facecolors='none',
                       marker=marker, s=20)
            if seg_type == 0 and plot_outlier_fraction is not None:
                i_min = int(plot_outlier_fraction*len(fx))
                i_max = int((1.-plot_outlier_fraction)*len(fx))
                fx_cut = np.sort(fx)[i_min:i_max]
                fy_cut = np.sort(fy)[i_min:i_max]
                fx_ext, fy_ext = [np.max(v)-np.min(v) for v in (fx_cut, fy_cut)]
                ax.set_xlim((np.min(fx_cut) - plot_f_edge*fx_ext,
                             np.max(fx_cut) + plot_f_edge*fx_ext))
                ax.set_ylim((np.min(fy_cut) - plot_f_edge*fy_ext,
                             np.max(fy_cut) + plot_f_edge*fy_ext))
            if i == n_features-1:
                ax.set_xlabel(feature_labels_used[j])
            if j == 0:
                ax.set_ylabel(feature_labels_used[i])

plt.show()
    
