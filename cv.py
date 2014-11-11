# Cross-validation functions.

import numpy as np

def cv_split_by_hour(X, hour_column=0, type_column=1):
    """
    Given a feature matrix X with hour indices in column hour_column
    and class (segment type) labels in column class_column, form a 
    cross-validation subsample by randomly selecting one set
    of preictal (seg_type=1) segments belonging to the same hour and
    1/N_hr_pre of the interictal (seg_type=0) segments (keeping segments
    in the same hour together), where N_hr_pre is the number of preictal
    hours (1/6 the number of preictal segments).
    Return the indices of the CV subsample and the remaining
    indices, which form the training subsample.
    """

    # get lists of unique preictal and interictal hour indices
    seg_type = X[:, type_column]
    pre_hrs = np.unique(X[seg_type == 1, hour_column])
    inter_hrs = np.unique(X[seg_type == 0, hour_column])

    # choose random hour indices for CV sample
    cv_pre_hrs = np.array([np.random.choice(pre_hrs)])
    cv_inter_hrs = np.random.choice(inter_hrs, len(inter_hrs) / len(pre_hrs),
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

