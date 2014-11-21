#!/usr/bin/env python

import sys
import os.path
from sklearn.linear_model import LogisticRegression
from train_model import train_model

submission_file = 'submission_log_reg_f1to3_rocslope10.csv'

features_files = [['data/Dog_1/features_02.txt'],
                  ['data/Dog_2/features_01.txt'],
                  ['data/Dog_3/features_01.txt'],
                  ['data/Dog_4/features_01.txt'],
                  ['data/Dog_5/features_01.txt'],
                  ['data/Patient_1/features_01.txt'],
                  ['data/Patient_2/features_01.txt']]
features_cols = [[8, 21], [12, 13], [3, 17], [11], [7, 10, 11], [10, 15], [5]]
C_vals = [0.01, 0.1, 1, 0.01, 10, 0.01, 10]

for ff, fc, C in zip(features_files, features_cols, C_vals):
    model, train_prob, test_prob = train_model(ff, fc, LogisticRegression,
                                            {'C': C,
                                            'class_weight': 'auto'},
                                            outlier_sigma=2, n_cv=100,
                                            normalize_probs='ROCSlope',
                                            plot=True, save_settings=True,
                                            submission_file=submission_file,
                                            verbose=True)

raw_input('Press Enter to end.')
