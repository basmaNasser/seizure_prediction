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

features_files = [os.path.abspath('data/Dog_1/features_02.txt')]
data_list_files = [os.path.abspath('data/Dog_1/features_02_data_files.txt')]
submission_file = os.path.abspath('submission_all_03.csv')
old_submission_file = os.path.abspath('sampeSubmission.csv')
default_prob = None    # default probability for submission
type_column = 1    # which column lists the segment type
hour_column = 0    # which column lists the hour index
n_cv = 100    # number of CV iterations
n_pre_hrs = 1    # number of 6-segment preictal clips to use in CV samples
n_learning_curve = 10    # number of steps for learning curves

feature_columns = [8]    # columns to include in model
C_reg = 0.001    # inverse of regularization strength

n_hr_col_pre_tot = 0
n_hr_col_inter_tot = 0
for i, f in enumerate(features_files):
    X_i = np.loadtxt(f)
    seg_type = X_i[:,type_column]
    n_hr_col_pre = len(np.unique(X_i[seg_type == 1,hour_column]))
    n_hr_col_inter = len(np.unique(X_i[seg_type == 0,hour_column]))
    X_i[seg_type == 1,hour_column] += n_hr_col_pre_tot
    X_i[seg_type == 0,hour_column] += n_hr_col_inter_tot
    if i == 0:
        X = np.copy(X_i)
    else:
        X = np.vstack((X, X_i))
    n_hr_col_pre_tot += n_hr_col_pre
    n_hr_col_inter_tot += n_hr_col_inter

X = features.scale_features(X)

model = linear_model.LogisticRegression(C=C_reg, class_weight='auto')

auc_values = []

# set up learning curve and ROC plots
fig = plt.figure(figsize=(8,4))
fig.set_tight_layout(True)
ax0 = plt.subplot(121)
ax1 = plt.subplot(122)

n_learn_avg = np.zeros(n_learning_curve)
cv_learn_avg = np.zeros(n_learning_curve)
train_learn_avg = np.zeros(n_learning_curve)

fp_rate_avg = np.linspace(0, 1, num=100)
tp_rate_avg = np.zeros(len(fp_rate_avg))

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
        # require both preictal and interictal classes to be present
        if len(np.unique(train_class[ix_train])) == 2:
            n_learn_avg[i_f] += 1
            n_train_array.append(n_train)
            model.fit(train_features[ix_train], train_class[ix_train])
            cv_learn.append(roc_auc_score(cv_class,
                                          model.predict_proba( \
                                              cv_features)[:,1]))
            cv_learn_avg[i_f] += cv_learn[-1]
            train_learn.append(roc_auc_score(train_class[ix_train],
                                             model.predict_proba( \
                                                train_features[ix_train])[:,1]))
            train_learn_avg[i_f] += train_learn[-1]
            
    ax0.plot(n_train_array, train_learn, linestyle='-', color=(1,0.6,0.6))
    ax0.plot(n_train_array, cv_learn, linestyle='-', color=(0.7,0.7,0.7))
    
    # train the model
    model.fit(train_features, train_class)
    print 'Feature coefficients:', model.coef_

    # get classification probabilities, plot ROC curve, and compute AUC
    p_pre = model.predict_proba(cv_features)[:,1]
    fp_rate, tp_rate, thresholds = roc_curve(cv_class, p_pre)
    tp_rate_avg += np.interp(fp_rate_avg, fp_rate, tp_rate)
    ax1.plot(fp_rate, tp_rate, linestyle='-', color=(0.7,0.7,0.7))
    auc = roc_auc_score(cv_class, p_pre)
    print 'AUC =', auc
    auc_values.append(auc)

    # compute AUC for training sample
    p_pre_train = model.predict_proba(train_features)[:,1]
    auc_train = roc_auc_score(train_class, p_pre_train)
    print 'training AUC =', auc_train

n_train_array = len(train_class)/float(n_learning_curve) * \
                    np.array(range(1, n_learning_curve+1))
ax0.plot(n_train_array, train_learn_avg/(n_learn_avg+1.e-3), 'r-',
         linewidth=3)
ax0.plot(n_train_array, cv_learn_avg/(n_learn_avg+1.e-3), 'k-',
         linewidth=3)
tp_rate_avg /= float(n_cv)
ax1.plot(fp_rate_avg, tp_rate_avg, 'k-', linewidth=3)

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
test_files = []
for lf in data_list_files:
    with open(lf, 'r') as df:
        data_files = df.readlines()
    for f in data_files:
        if 'test' in f:
            test_files.append(f.strip())

# update submission file
submission.update_submission(dict(zip(test_files, p_pre_test)),
                             submission_file, default_value=default_prob,
                             old_submission_file=old_submission_file)

# show plot
ax0.set_ylim((0.5, 1))
ax0.set_xlabel('number of training instances')
ax0.set_ylabel('AUC') 
ax1.plot(np.linspace(0, 1), np.linspace(0, 1), 'k:')
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
plt.show()
