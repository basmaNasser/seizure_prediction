#!/usr/bin/env python
#
# Train a logistic regression model with cross-validation samples.

import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score
import cv
import features
import submission

features_file = os.path.abspath('data/Dog_1/features_01.txt')
data_list_file = os.path.abspath('data/Dog_1/features_01_data_files.txt')
submission_file = os.path.abspath('submission_Dog_1_01.csv')
type_column = 1    # column listing segment type
n_cv = 10    # number of CV iterations
n_pre_hrs = 1    # number of 6-segment preictal clips to use in CV samples
n_learning_curve = 20    # number of steps for learning curves

feature_columns = [5]    # columns to include in model
C_reg = 0.1    # inverse of regularization strength

X = np.loadtxt(features_file)
X = features.scale_features(X)

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
    print '\nCV iteration', i_cv+1
    indices = cv.cv_split_by_hour(X, n_pre_hrs=n_pre_hrs)

    # get feature matrices and class arrays for training and CV samples
    train_features_all, cv_features_all = [X[indices[k],:] \
                                           for k in ['train', 'cv']]
    train_features, cv_features = [y[:,np.array(feature_columns)] for y in \
                                   [train_features_all, cv_features_all]]
    train_class, cv_class = [X[indices[k], type_column] \
                             for k in ['train', 'cv']]

    pct_train = 100.*len(train_class)/float(len(train_class)+len(cv_class))
    print len(train_class), 'training instances ({0:.1f}%)'.format(pct_train)
    print len(cv_class), 'CV instances ({0:.1f}%)'.format(100.-pct_train)

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
    print 'Feature coefficients:', model.coef_

    # get classification probabilities, plot ROC curve, and compute AUC
    p_pre = model.predict_proba(cv_features)[:,1]
    fp_rate, tp_rate, thresholds = roc_curve(cv_class, p_pre)
    ax1.plot(fp_rate, tp_rate, 'k-')
    auc = roc_auc_score(cv_class, p_pre)
    print 'AUC =', auc
    auc_values.append(auc)

    # compute AUC for training sample
    p_pre_train = model.predict_proba(train_features)[:,1]
    auc_train = roc_auc_score(train_class, p_pre_train)
    print 'training AUC =', auc_train
    
print '\n Average AUC:'
print np.mean(auc_values), '+/-', np.std(auc_values)

# re-train on entire training sample
train_features_all = X[(X[:,type_column] == 0) | (X[:,type_column] == 1),:]
train_features = train_features_all[:,np.array(feature_columns)]
train_class = train_features_all[:,type_column]
model.fit(train_features, train_class)

# predict probabilities for test data
test_features_all = X[X[:,type_column] == -1,:]
test_features = test_features_all[:,np.array(feature_columns)]
p_pre_test = model.predict_proba(test_features)[:,1]

# get test file names
with open(data_list_file, 'r') as df:
    data_files = df.readlines()
test_files = []
for f in data_files:
    if 'test' in f:
        test_files.append(f.strip())

# update submission file
submission.update_submission(dict(zip(test_files, p_pre_test)),
                             submission_file)

# show plot
ax0.set_xlabel('number of training instances')
ax0.set_ylabel('AUC') 
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
plt.show()