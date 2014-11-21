#!/usr/bin/env python

import sys
import os.path
from sklearn.svm import SVC

sys.path.insert(0, '.')
from train_model import train_model

submission_file = 'submission_svm_f1to2_rocslope5.csv'

features_files = [['data/Dog_1/features_02.txt'],
                  ['data/Dog_2/features_01.txt'],
                  ['data/Dog_3/features_01.txt'],
                  ['data/Dog_4/features_01.txt'],
                  ['data/Dog_5/features_01.txt'],
                  ['data/Patient_1/features_01.txt'],
                  ['data/Patient_2/features_01.txt']]
features_cols = [[5, 11], [13], [3, 7], [5, 11], [14], [4, 11], [17, 18]]
C_vals = [10, 10, 0.1, 0.1, 0.1, 10, 10]
gamma_vals = [1, 0.5, 0.25, 0.25, 1, 1, 1]

for ff, fc, C, gamma in zip(features_files, features_cols, C_vals, gamma_vals):
    model, train_prob, test_prob = train_model(ff, fc, SVC,
                                            {'C': C, 'kernel': 'rbf',
                                            'gamma': gamma,
                                            'probability': True,
                                            'class_weight': 'auto'},
                                            outlier_sigma=2, n_cv=10,
                                            normalize_probs='ROCSlope',
                                            plot=True, save_settings=True,
                                            submission_file=submission_file,
                                            verbose=True)

raw_input('Press Enter to end.')
