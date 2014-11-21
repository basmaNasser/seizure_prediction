#!/usr/bin/env python

import sys
import os.path
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, '.')
from train_model import train_model

submission_file = 'submission_rf_f4imp_all.csv'

features_files = ['data/Dog_1/features_02.txt',
                  'data/Dog_2/features_01.txt',
                  'data/Dog_3/features_01.txt',
                  'data/Dog_4/features_01.txt',
                  'data/Dog_5/features_01.txt',
                  'data/Patient_1/features_01.txt',
                  'data/Patient_2/features_01.txt']

model, train_prob, test_prob = train_model(features_files, [11, 5, 4, 6],
                                           RandomForestClassifier,
                                           {'n_estimators': 50,
                                            'criterion': 'entropy',
                                            'min_samples_leaf': 10},
                                           outlier_sigma=2, n_cv=20,
                                           plot=True, save_settings=True,
                                           submission_file=submission_file,
                                           verbose=True)

raw_input('Press Enter to end.')
