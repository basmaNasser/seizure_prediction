#!/usr/bin/env python
#
# Train a logistic regression model with cross-validation samples.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import cv
import features
import submission

def predict_probs(model, train_class, train_features, test_features,
                  normalize_probs=False):
    """
    Fit a given binary classification model to training sample features
    and return predicted probabilities for the positive class for
    the training and test samples.
    """
    model.fit(train_features, train_class)
    train_prob, test_prob = [model.predict_proba(f)[:,1] for f in \
                                    (train_features, test_features)]
    if normalize_probs == 'IsoReg':
        prob_model = IsotonicRegression(out_of_bounds='clip')
        prob_model.fit(train_prob, train_class)
        train_prob, test_prob = [prob_model.transform(p) for p in \
                                        (train_prob, test_prob)]
    return (train_prob, test_prob)
        

def learning_curve(model, train, test, n=10, metric=roc_auc_score, **kwargs):
    """
    Compute learning curves (accuracy or other evaluation metric for training
    and test samples vs. number of training instances) for a given model.
    """
    learn_test = []
    learn_train = []
    n_train_array = []
    mask = []
    # shuffle indices
    ix_train_all = np.random.choice(len(train[1]), len(train[1]),
                                    replace=False)
    for i in range(n):
        n_train = int(len(train[1]) * (i+1) / float(n))
        ix_train = ix_train_all[:n_train]
        # require both preictal and interictal classes to be present
        if len(np.unique(train[1][ix_train])) == 2:
            train_prob, test_prob = predict_probs(model, train[1][ix_train], \
                                      train[0][ix_train], \
                                      test[0], **kwargs)
            if np.any(np.isnan(test_prob)) or np.any(np.isnan(train_prob)):
                continue
            learn_test.append(metric(test[1], test_prob))
            learn_train.append(metric(train[1][ix_train], train_prob))
            n_train_array.append(n_train)
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    return (mask, n_train_array, learn_train, learn_test)


def train_model(features_files, feature_columns, classifier, model_args,
                outlier_sigma=None, scale_features=True,
                submission_file=None, save_settings=False, plot=False,
                normalize_probs=False, n_cv=10, f_cv=0.3, verbose=False):
    """
    Fit a classification model (classifier, using arguments in model_args)
    to the features in columns feature_columns in the file(s) in
    features_files. Use CV with n_cv random training-CV sample splittings,
    each containing a fraction f_cv in the CV subsample, to estimate AUC
    for the fit.
    """
    settings = locals()
    hour_column = 0
    type_column = 1

    # read in feature matrix from file(s)
    X = features.load_features(features_files)
    # remove outliers
    if outlier_sigma is not None:
        X, retained_indices = features.remove_outliers(X, n_sigma=outlier_sigma)
    # scale features
    if scale_features:
        X = features.scale_features(X)

    # set up model
    model = classifier(**model_args)

    # set up plot
    if plot:
        fig = plt.figure(figsize=(8,4))
        fig.set_tight_layout(True)
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)
        # initialize plot arrays
        n_learn = np.zeros(10)
        learn_cv_avg = np.zeros(len(n_learn))
        learn_train_avg = np.zeros(len(n_learn))
        fp_rate_avg = np.linspace(0, 1, num=100)
        tp_rate_avg = np.zeros(len(fp_rate_avg))
    
    # loop over training-CV sample splittings
    auc_values = []
    for i_cv in range(n_cv):
        cv_indices = cv.cv_split_by_hour(X, n_pre_hrs=f_cv)
        if verbose:
            print '\nCV iteration', i_cv+1
            print len(cv_indices['train']), 'training instances'
            print len(cv_indices['cv']), 'CV instances'
        # get feature matrices and class arrays for training and CV samples
        train_features_all, cv_features_all = [X[cv_indices[k],:] \
                                               for k in ['train', 'cv']]
        train_features, cv_features = [y[:,np.array(feature_columns)] for y in \
                                       [train_features_all, cv_features_all]]
        train_class = train_features_all[:,type_column]
        cv_class = cv_features_all[:,type_column]

        # compute learning curve
        if plot:
            learn_mask, n_train, learn_train, learn_cv = \
                    learning_curve(model, (train_features, train_class),
                                   (cv_features, cv_class), n=len(n_learn),
                                   normalize_probs=normalize_probs)
            n_learn[learn_mask] += 1
            learn_train_avg[learn_mask] += learn_train
            learn_cv_avg[learn_mask] += learn_cv
            ax0.plot(n_train, learn_train, linestyle='-', color=(1,0.6,0.6))
            ax0.plot(n_train, learn_cv, linestyle='-', color=(0.7,0.7,0.7))

        # predict probabilities
        train_prob, cv_prob = predict_probs(model, train_class, train_features,
                                            cv_features, normalize_probs)
        if verbose:
            print 'Feature coefficients:', model.coef_

        # compute AUC
        auc = roc_auc_score(cv_class, cv_prob)
        auc_values.append(auc)
        if verbose:
            print 'training AUC =', roc_auc_score(train_class, train_prob)
            print 'CV AUC =', auc

        # plot ROC curve
        if plot:
            fp_rate, tp_rate, thresholds = roc_curve(cv_class, cv_prob)
            tp_rate_avg += np.interp(fp_rate_avg, fp_rate, tp_rate)
            ax1.plot(fp_rate, tp_rate, linestyle='-', color=(0.7,0.7,0.7))
            

    # compute mean and std. dev. of AUC over CV iterations
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values)
    if verbose:
        print '\nAverage AUC:', auc_mean, '+/-', auc_std

    # update submission CSV file
    if submission_file is not None:
        train_features_all = X[(X[:,type_column] == 0) | \
                               (X[:,type_column] == 1),:]
        train_features = train_features_all[:,np.array(feature_columns)]
        train_class = train_features_all[:,type_column]
        test_features_all = X[X[:,type_column] == -1,:]
        test_features = test_features_all[:,np.array(feature_columns)]
        train_prob, test_prob = predict_probs(model, train_class,
                                              train_features, test_features,
                                              normalize_probs)
        test_files = []
        for ff in features_files:
            data_list_file = '.'.join(ff.split('.')[:-1]) + '_data_files.txt'
            with open(data_list_file, 'r') as df:
                data_files = np.array(df.readlines())
                if outlier_sigma is not None:
                    data_files = data_files[retained_indices]
            for f in data_files:
                if 'test' in f:
                    test_files.append(f.strip())
        submission.update_submission(dict(zip(test_files, test_prob)),
                                     submission_file)

    # save settings
    if save_settings:
        if submission_file is not None:
            settings_file = '.'.join(submission_file.split('.')[:-1]) + \
                                '_settings.txt'
            open_mode = 'a'
        else:
            settings_file = 'train_model_settings.txt'
            open_mode = 'w'
        with open(settings_file, open_mode) as sf:
            for s in ['features_files', 'feature_columns', 'classifier',
                      'model_args', 'outlier_sigma', 'scale_features',
                      'submission_file', 'normalize_probs']:
                if s in settings:
                    sf.write(s + ': ' + str(settings[s]) + '\n')
            sf.write('AUC = {0:.2f}+/-{1:.2f}\n\n'.format(auc_mean, auc_std))

    # plot average learning curves and ROC curve
    if plot:
        n_train_array = len(cv_indices['train'])/float(len(n_learn)) * \
                        np.array(range(1, len(n_learn)+1))
        ax0.plot(n_train_array, learn_train_avg/(n_learn+1.e-3), 'r-',
                 linewidth=3)
        ax0.plot(n_train_array, learn_cv_avg/(n_learn+1.e-3), 'k-',
                 linewidth=3)
        tp_rate_avg /= float(n_cv)
        ax1.plot(fp_rate_avg, tp_rate_avg, 'k-', linewidth=3)
        # display plot
        ax0.set_ylim((0.5, 1))
        ax0.set_xlabel('number of training instances')
        ax0.set_ylabel('AUC') 
        ax1.plot(np.linspace(0, 1), np.linspace(0, 1), 'k:', linewidth=2)
        ax1.set_xlabel('false positive rate')
        ax1.set_ylabel('true positive rate')
        plt.show(block=False)

    return (model, auc_mean, auc_std)

