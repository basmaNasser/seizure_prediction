#!/usr/bin/env python

import sys
import os.path
from optimize_model import optimize_model

submission_file = 'submission_log_reg_f1to4_reopt_rocslope10.csv'

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
    optimize_model(f, submission_file, max_features=4,
                   outlier_sigma=2, normalize_probs='ROCSlope', n_cv=100)

raw_input('Press Enter to end.')
