#!/usr/bin/env python
#
# Perform a grid search in several settings for training a
# classification model to optimize the AUC.

import sys
import itertools
import numpy as np
from sklearn import linear_model
from train_model import train_model

def optimize_model(features_files, submission_file,
                   classifier=linear_model.LogisticRegression,
                   feature_columns=range(2, 26),
                   min_features=1, max_features=1,
                   parameters={'C': np.logspace(-3, 1, 5),
                               'class_weight': ['auto']},
                   **kwargs):
    """
    Find the combination of features (selected from columns listed
    in feature_columns in the files listed in features_files,
    including 1 to max_features different features) and model
    parameters that optimizes the AUC for the chosen classifier,
    then update predicted test probabilities in submission_file.
    In the parameters dictionary, the keys are the keywords to be used
    when initializing the classifier, and each value is an array of
    arguments to loop over when searching for the best model.
    Additional settings are passed to train_model in kwargs.
    """
    best_model = {'AUC': 0.}
    
    # vary number of features
    if min_features > max_features:
        sys.exit('min_features must be <= max_features')
    for n_features in range(min_features, max_features+1):
        # vary features used in model training
        if n_features == min_features:
            if min_features == 1:
                feature_columns_grid = [[i] for i in feature_columns]
            else:
                feature_columns_grid = list(itertools.combinations( \
                                           feature_columns, min_features))
        else:
            remaining_features = list(feature_columns)
            for i in best_model['columns']:
                remaining_features.remove(i)
            feature_columns_grid = [list(best_model['columns']) + [i] \
                                    for i in remaining_features]
        for f_cols in feature_columns_grid:
            # vary model parameters
            for model_args in list(itertools.product( \
                                  *[[(k, v) for v in parameters[k]] \
                                  for k in parameters.keys()])):
                # train the classifier and compute AUC
                model, auc_mean, auc_std = train_model(features_files,
                                                       f_cols, classifier,
                                                       dict(model_args),
                                                       **kwargs)
                print '\r' + ', '.join([str(fc) for fc in f_cols]) + \
                        ' AUC = {0:.2f}+/-{1:.2f}'.format(auc_mean, auc_std),
                sys.stdout.flush()
                if auc_mean > best_model['AUC']:
                    print '\n    ', model_args
                    best_model = {'AUC': auc_mean,
                                  'columns': f_cols,
                                  'parameters': model_args}
    print '\r' + 70*' ' + '\n'

    # compute predictions for best model, update submission CSV,
    # and plot learning curves and ROC curve
    model, auc_mean, auc_std = train_model(features_files, 
                                           best_model['columns'], classifier,
                                           dict(best_model['parameters']),
                                           submission_file=submission_file,
                                           save_settings=True, plot=True,
                                           **kwargs)

