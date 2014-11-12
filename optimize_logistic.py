#!/usr/bin/env python
#
# Perform a grid search in several settings for training a
# logistic regression model to optimize the AUC.

import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score
import cv
import features

# basic settings (fixed)
features_file = os.path.abspath('data/Dog_1/features_01.txt')
type_column = 1    # column listing segment type
n_cv = 50    # number of CV iterations
n_pre_hrs = 1    # number of 6-segment preictal clips to use in CV samples
n_learning_curve = 20    # number of steps for learning curves

# lists of settings for grid search
# columns to include in model
feature_columns_grid = ([2], [3], [4], [5], [6], [4, 5], [2, 3, 4, 5, 6])
# inverse of regularization strength
C_reg_grid = np.logspace(-3, 1, 13)

X = np.loadtxt(features_file)
X = features.scale_features(X)

avg_auc_best = 0.

for feature_columns, C_reg in itertools.product(feature_columns_grid,
                                                C_reg_grid):
    print '\nFeatures used:', feature_columns
    print 'C =', C_reg

    model = linear_model.LogisticRegression(C=C_reg, class_weight='auto')

    auc_values = []

    # loop over random training-CV sample splittings
    for i_cv in range(n_cv):
        indices = cv.cv_split_by_hour(X, n_pre_hrs=n_pre_hrs)

        # get feature matrices and class arrays for training and CV samples
        train_features_all, cv_features_all = [X[indices[k],:] \
                                               for k in ['train', 'cv']]
        train_features, cv_features = [y[:,np.array(feature_columns)] for y in \
                                       [train_features_all, cv_features_all]]
        train_class, cv_class = [X[indices[k], type_column] \
                                 for k in ['train', 'cv']]

        # train the model
        model.fit(train_features, train_class)

        # get classification probabilities and compute AUC
        p_pre = model.predict_proba(cv_features)[:,1]
        auc = roc_auc_score(cv_class, p_pre)
        auc_values.append(auc)

    # compute average AUC from CV iterations and compare to best value
    avg_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    print 'AUC =', avg_auc, '+/-', std_auc
    if avg_auc > avg_auc_best:
        avg_auc_best = avg_auc
        std_auc_best = std_auc
        feature_columns_best = feature_columns
        C_reg_best = C_reg


# report best values and retrain with those settings
print '\n---------------'
print 'Best AUC =', avg_auc_best, '+/-', std_auc_best
print 'Features:', feature_columns_best
print 'C =', C_reg_best

print '\nRetraining with optimal settings...'

feature_columns = feature_columns_best
C_reg = C_reg_best
model = linear_model.LogisticRegression(C=C_reg, class_weight='auto')
auc_values = []

# set up learning curve and ROC plots
fig = plt.figure(figsize=(8,4))
fig.set_tight_layout(True)
ax0 = plt.subplot(121)
ax1 = plt.subplot(122)
ax1.plot(np.linspace(0, 1), np.linspace(0, 1), 'k:')

# loop over random training-CV sample splittings
for i_cv in range(n_cv):
    indices = cv.cv_split_by_hour(X, n_pre_hrs=n_pre_hrs)

    # get feature matrices and class arrays for training and CV samples
    train_features_all, cv_features_all = [X[indices[k],:] \
                                           for k in ['train', 'cv']]
    train_features, cv_features = [y[:,np.array(feature_columns)] for y in \
                                   [train_features_all, cv_features_all]]
    train_class, cv_class = [X[indices[k], type_column] \
                             for k in ['train', 'cv']]

    # loop over fractions of training data to compute learning curve
    cv_learn = []
    train_learn = []
    n_train_array = []
    # shuffle indices
    ix_train_all = np.random.choice(len(train_class), len(train_class),
                                    replace=False)
    for i_f in range(n_learning_curve):
        n_train = int(len(train_class) * (i_f+1) / float(n_learning_curve))
        ix_train = ix_train_all[:n_train]
        if 1 in train_class[ix_train]: # require at least 1 preictal case
            n_train_array.append(n_train)
            model.fit(train_features[ix_train], train_class[ix_train])
            cv_learn.append(roc_auc_score(cv_class,
                                          model.predict_proba( \
                                              cv_features)[:,1]))
            train_learn.append(roc_auc_score(train_class[ix_train],
                                             model.predict_proba( \
                                                train_features[ix_train])[:,1]))
    ax0.plot(n_train_array, train_learn, 'r-')
    ax0.plot(n_train_array, cv_learn, 'k-')
    
    # train the model
    model.fit(train_features, train_class)

    # get classification probabilities, plot ROC curve, and compute AUC
    p_pre = model.predict_proba(cv_features)[:,1]
    fp_rate, tp_rate, thresholds = roc_curve(cv_class, p_pre)
    ax1.plot(fp_rate, tp_rate, 'k-')
    auc = roc_auc_score(cv_class, p_pre)
    auc_values.append(auc)
    
print '\nAverage AUC:'
print np.mean(auc_values), '+/-', np.std(auc_values)

ax0.set_xlabel('number of training instances')
ax0.set_ylabel('AUC') 
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
plt.show()
