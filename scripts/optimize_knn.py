#!/usr/bin/env python

import sys
import os.path
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, '.')
from optimize_model import optimize_model

submission_file = 'submission_knn_f3.csv'

features_files = [['data/Dog_1/features_02.txt']]
"""
features_files = [['data/Dog_1/features_02.txt',
                  'data/Dog_2/features_01.txt',
                  'data/Dog_3/features_01.txt',
                  'data/Dog_4/features_01.txt',
                  'data/Dog_5/features_01.txt',
                  'data/Patient_1/features_01.txt',
                  'data/Patient_2/features_01.txt']]
"""

for f in features_files:
    if type(f) == str:
        sys.exit('Each element of features_files must be a list of files.')
    print f
    optimize_model(f, submission_file, min_features=1, max_features=2,
                   classifier=KNeighborsClassifier,
                   parameters={'n_neighbors': [5, 10, 15],
                               'weights': ['uniform', 'distance']},
                   outlier_sigma=2, normalize_probs=None, n_cv=100)

raw_input('Press Enter to end.')
