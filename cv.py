# Cross-validation functions.

import sys
import numpy as np

def cv_split_by_hour(X, hour_column=0, type_column=1, n_pre_hrs=1):
    """
    Given a feature matrix X with hour indices in column hour_column
    and class (segment type) labels in column class_column, form a 
    cross-validation subsample by randomly selecting n_pre_hrs
    hour-long clips of preictal (seg_type=1) segments and
    n_pre_hrs/N of the interictal (seg_type=0) segments (keeping segments
    in the same hour together), where N is the total number of preictal
    hours (1/6 the number of preictal segments).
    If n_pre_hrs < 1, interpret it as the approximate fraction of
    preictal hours in the CV subsample instead (n_pre_hrs/N).
    Return the indices of the CV subsample and the remaining
    indices, which form the training subsample.
    """

    # get lists of unique preictal and interictal hour indices
    # (only including hours with all 6 segments)
    seg_type = X[:, type_column]
    pre_hrs, pre_hr_count = np.unique(X[seg_type == 1, hour_column],
                                      return_counts=True)
    pre_hrs = pre_hrs[pre_hr_count % 6 == 0]
    inter_hrs, inter_hr_count = np.unique(X[seg_type == 0, hour_column],
                                          return_counts=True)
    inter_hrs = inter_hrs[inter_hr_count % 6 == 0]

    # choose random hour indices for CV sample
    if n_pre_hrs >= len(pre_hrs):
        sys.exit('Too many preictal hours for CV sample.')
    if n_pre_hrs < 1:
        n_pre_hrs = np.max([int(n_pre_hrs*len(pre_hrs)), 1])
    if n_pre_hrs == 1:
        cv_pre_hrs = np.array([np.random.choice(pre_hrs)])
    else:
        cv_pre_hrs = np.random.choice(pre_hrs, n_pre_hrs, replace=False)
    cv_inter_hrs = np.random.choice(inter_hrs,
                                    len(inter_hrs)*int(n_pre_hrs)/len(pre_hrs),
                                    replace=False)

    # find row indices for CV sample
    hour = X[:, hour_column]
    cv_pre_indices = np.where((seg_type == 1) & [hour[i] in cv_pre_hrs \
                                              for i in range(len(hour))])[0]
    cv_inter_indices = np.where((seg_type == 0) & [hour[i] in cv_inter_hrs \
                                                for i in range(len(hour))])[0]
    cv_indices = np.concatenate((cv_pre_indices, cv_inter_indices))

    # find row indices for training sample
    all_indices = set(np.where((seg_type == 0) | (seg_type == 1))[0])
    train_indices = np.array(list(all_indices.difference(set(cv_indices))))

    return {'train': train_indices, 'cv': cv_indices}

