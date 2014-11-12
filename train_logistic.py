#!/usr/bin/env python
#
# Train a logistic regression model with cross-validation samples.

import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score
import cv

features_file = os.path.abspath('data/Dog_1/features_01.txt')
type_column = 1    # column listing segment type
feature_columns = [2, 3, 4, 5]    # columns to include in model
n_cv = 10    # number of CV iterations
n_pre_hrs = 1    # number of 6-segment preictal clips to use in CV samples
C_reg = 1.0    # inverse of regularization strength

features = np.loadtxt(features_file)

model = linear_model.LogisticRegression(C=C_reg, class_weight='auto')

auc_values = []

# set up ROC plot
fig = plt.figure(figsize=(6,4))
fig.set_tight_layout(True)
plt.plot(np.linspace(0, 1), np.linspace(0, 1), 'k:')

# loop over random training-CV sample splittings
for i_cv in range(n_cv):
    print '\nCV iteration', i_cv+1
    indices = cv.cv_split_by_hour(features, n_pre_hrs=n_pre_hrs)

    # get feature matrices and class arrays for training and CV samples
    train_features_all, cv_features_all = [features[indices[x],:] \
                                           for x in ['train', 'cv']]
    train_features, cv_features = [x[:,np.array(feature_columns)] for x in \
                                   [train_features_all, cv_features_all]]
    train_class, cv_class = [features[indices[x], type_column] \
                             for x in ['train', 'cv']]

    pct_train = 100.*len(train_class)/float(len(train_class)+len(cv_class))
    print len(train_class), 'training instances ({0:.1f}%)'.format(pct_train)
    print len(cv_class), 'CV instances ({0:.1f}%)'.format(100.-pct_train)
                             
    # train the model
    model.fit(train_features, train_class)
    print 'Feature coefficients:', model.coef_

    # get classification probabilities, plot ROC curve, and compute AUC
    p_pre = model.predict_proba(cv_features)[:,1]
    fp_rate, tp_rate, thresholds = roc_curve(cv_class, p_pre)
    plt.plot(fp_rate, tp_rate, 'k-')
    auc = roc_auc_score(cv_class, p_pre)
    print 'AUC =', auc
    auc_values.append(auc)

print '\n Average AUC:'
print np.mean(auc_values), '+/-', np.std(auc_values)
    
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
