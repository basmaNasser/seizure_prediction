#!/usr/bin/env python

import sys
import os.path
from sklearn.svm import SVC

sys.path.insert(0, '.')
from optimize_model import optimize_model

submission_file = 'submission_svm_f1to2.csv'

features_files = [['data/Dog_1/features_02.txt'],
                  ['data/Dog_2/features_01.txt'],
                  ['data/Dog_3/features_01.txt'],
                  ['data/Dog_4/features_01.txt'],
                  ['data/Dog_5/features_01.txt'],
                  ['data/Patient_1/features_01.txt'],
                  ['data/Patient_2/features_01.txt']]

for f in features_files:
    if type(f) == str:
        sys.exit('Each element of features_files must be a list of files.')
    print f
    optimize_model(f, submission_file, min_features=1, max_features=2,
                   feature_columns=range(2, 27),
                   classifier=SVC,
                   parameters={'C': [0.1, 1, 10],
                               'gamma': [0.25, 0.5, 1],
                               'kernel': ['rbf'], 'probability': [True],
                               'class_weight': ['auto']},
                   outlier_sigma=2, normalize_probs='IsoReg', n_cv=20)

raw_input('Press Enter to end.')
