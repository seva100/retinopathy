'''
Retinopathy project.
Auxiliary functions.

@author: Artem Sevastopolsky, 2016
'''

from functools import partial
import itertools
import numpy as np
import pandas as pd
import sklearn
import sklearn.cross_validation


def dict_subset(d, keys):
    subset = tuple(d[k] for k in keys)
    return subset


def grid_search(func, kwargs, result_keys):
    '''
    Performs grid search for function over combinations of kwargs.
    - kwargs should be a dict where keys are parameters and values are lists of values of these parameters.
    Remarks:
    1. it's important to store values in lists, not other array-like-s
    2. if there's only one value for a parameter, store it in list too.
    - It is assumed that function returns a dict. result_keys is an iterable of keys of returned dict
    to present in columns of resulting DataFrame.
    Example:
    >>> grid_search(svm_libsvm_solver, 
                    {'C': [1, 5, 100],
                     'gamma': [1],
                     'X_test': [<array ...>],
                     ....},
                     ('CV error', 'Iter num',))
    '''
    
    args_keys, args_vals = list(kwargs.keys()), list(kwargs.values())
    iter_keys = [k for k in args_keys if len(kwargs[k]) > 1]
    iter_vals = dict_subset(kwargs, iter_keys)
    df = pd.DataFrame(index=pd.MultiIndex.from_product(iter_vals, names=iter_keys),
                      columns=result_keys)
    
    for comb in itertools.product(*args_vals):
        comb_dict = dict(zip(args_keys, comb))
        comb_iter_vals = dict_subset(comb_dict, iter_keys)
        #print(comb_iter_vals)
        ans_dict = func(**comb_dict)
        #print('comb: {}, ans: {}'.format(comb, ans))
        for k in result_keys:
            df.ix[comb_iter_vals, k] = ans_dict[k]
    return df


def get_true_and_pred_CV(estimator, X, y, n_folds, cv, params):
    ys = []
    for train_idx, valid_idx in cv:
        clf = estimator(**params)
        #print train_idx
        if isinstance(X, np.ndarray):
            clf.fit(X[train_idx], y[train_idx])
            cur_pred = clf.predict(X[valid_idx])
        elif isinstance(X, pd.DataFrame):
            clf.fit(X.iloc[train_idx, :], y[train_idx]) 
            cur_pred = clf.predict(X.iloc[valid_idx, :])
        else:
            raise Exception('Only numpy array and pandas DataFrame ' \
                            'as types of X are supported')
        
        ys.append((y[valid_idx], cur_pred))
    return ys


def fit_and_score_CV(estimator, X, y, n_folds=10, stratify=True, **params):
    #algo = estimator(**params)
    #print(algo)
    if not stratify:
        cv_arg = sklearn.cross_validation.KFold(y.size, n_folds)
    else:
        cv_arg = sklearn.cross_validation.StratifiedKFold(y, n_folds)
    
    ys = get_true_and_pred_CV(estimator, X, y, n_folds, cv_arg, params)    
    cv_acc = map(lambda tp: sklearn.metrics.accuracy_score(tp[0], tp[1]), ys)
    #cv_pr_weighted = map(lambda tp: sklearn.metrics.precision_score(tp[0].astype(float), tp[1].astype(float), average='weighted'), ys)
    #cv_rec_weighted = map(lambda tp: sklearn.metrics.recall_score(tp[0].astype(float), tp[1].astype(float), average='weighted'), ys)
    #cv_f1_weighted = map(lambda tp: sklearn.metrics.f1_score(tp[0].astype(float), tp[1].astype(float), average='weighted'), ys)
    cv_conf_mx = map(lambda tp: sklearn.metrics.confusion_matrix(tp[0], tp[1]), ys)
    # specificity = tn / (tn + fp)
    # sensitivity = tp / (tp + fn)
    # confusion matrix:
    # TP FN
    # FP TN
    cv_specificity = map(lambda conf_mx: float(conf_mx[1, 1]) / (conf_mx[1, 1] + conf_mx[1, 0] + 1e-8), cv_conf_mx)
    cv_sensitivity = map(lambda conf_mx: float(conf_mx[0, 0]) / (conf_mx[0, 0] + conf_mx[0, 1] + 1e-8), cv_conf_mx)
    cv_specificity = np.array(cv_specificity)
    cv_sensitivity = np.array(cv_sensitivity)
    cv_f1_spec_sens = (2.0 * cv_specificity * cv_sensitivity) / \
        (cv_specificity + cv_sensitivity).astype(float)
    # the approach below makes estimator fit multiple times
    #cv_acc = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='accuracy')
    #cv_pr_weighted = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='precision_weighted')
    #cv_rec_weighted = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='recall_weighted')   
    #cv_f1_weighted = sklearn.cross_validation.cross_val_score(algo, X, y, cv=cv_arg, scoring='f1_weighted') 
    #print(cv_score)
    return {'CV accuracy': np.mean(cv_acc),
            'CV specificity': np.mean(cv_specificity), 'CV sensitivity': np.mean(cv_sensitivity), 
            'CV F1 of spec. and sens.': np.mean(cv_f1_spec_sens)}


def grid_fit_search_CV(estimator, params_grid, X, y, n_folds=10, stratify=True):
    gs_result = grid_search(partial(fit_and_score_CV, estimator=estimator, 
                                    X=X, y=y, n_folds=n_folds, stratify=stratify), 
                            params_grid, ['CV accuracy', 
                                          'CV specificity', 'CV sensitivity',
                                          'CV F1 of spec. and sens.'])
    return gs_result