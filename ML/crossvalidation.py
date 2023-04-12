import time
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from afml.cross_validation.cross_validation import ml_cross_val_score
from afml.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def get_clf_best_param_cv(type, clf, X_train, y_train, cv_gen, scoring, sample_weight, scaler=StandardScaler()):
    t0 = 0.0
    t1 = 0.0

    best_param_dict = {}
    best_param_dict['type'] = type
    best_param_dict['best_model'] = None
    best_param_dict['best_cross_val_score'] = -np.inf
    best_param_dict['recall'] = -np.inf
    best_param_dict['precision'] = -np.inf
    best_param_dict['accuracy'] = -np.inf
    best_param_dict['run_time'] = 0.0

    col = X_train.columns.to_list()
    idx = X_train.index

    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=col, index=idx)

    t0 = time.time()
    temp_score_base, temp_recall, temp_precision, temp_accuracy = ml_cross_val_score(clf, X_train_scaled, y_train, cv_gen, scoring=scoring, sample_weight=sample_weight)
    t1 = time.time()
    
    if temp_score_base.mean() > best_param_dict['best_cross_val_score']:
    #if temp_recall.mean() > best_param_dict['recall']:
        best_param_dict['best_model'] = clf
        best_param_dict['best_cross_val_score'] = temp_score_base.mean()
        best_param_dict['recall'] = temp_recall.mean()
        best_param_dict['precision'] = temp_precision.mean()
        best_param_dict['accuracy'] = temp_accuracy.mean()
        best_param_dict['run_time'] = t1-t0    
    
    return best_param_dict

def perform_grid_search(X_train, y_train, cv_gen, scoring, parameters, events, dollar_bars, type='sequential_bootstrapping_SVC', sample_weight=None, RANDOM_STATE=42):
    """
    Grid search using Purged CV without using sample weights in fit(). Returns top model and top score
    """

    if type=='SVC' or type=='sequential_bootstrapping_SVC':
        for C in parameters['C']:
            for gamma in parameters['gamma']:

                clf_SVC = SVC(C=C,
                                gamma=gamma,
                                class_weight='balanced',
                                kernel='linear',
                                random_state=RANDOM_STATE)

                if type =='SVC':
                    clf = clf_SVC
                elif type == 'sequential_bootstrapping_SVC':
                    clf = SequentiallyBootstrappedBaggingClassifier(samples_info_sets=events.loc[X_train.index].t1, ## events
                                                                price_bars = dollar_bars.loc[X_train.index.min():X_train.index.max(), 'close'], ## df
                                                                estimator=clf_SVC, 
                                                                random_state=RANDOM_STATE, n_jobs=-1, oob_score=False,
                                                                max_features=1.)

                # get best param dict   
                best_param_dict = get_clf_best_param_cv(type, clf, X_train, y_train, cv_gen, scoring=scoring, sample_weight=sample_weight)


    else:    
        for m_depth in parameters['max_depth']:
            for n_est in parameters['n_estimators']:
                clf_base = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, 
                                                max_depth=m_depth, class_weight='balanced')

                if type == 'standard_bagging_decision_tree':
                    clf = BaggingClassifier(n_estimators=n_est, 
                                            estimator=clf_base, 
                                            random_state=RANDOM_STATE, n_jobs=-1, 
                                            oob_score=False, max_features=1.)
                elif type == 'random_forest':
                    clf = RandomForestClassifier(n_estimators=n_est, 
                                                max_depth=m_depth, 
                                                random_state=RANDOM_STATE, 
                                                n_jobs=-1, 
                                                oob_score=False, 
                                                criterion='entropy',
                                                class_weight='balanced_subsample', 
                                                max_features=1.)
                elif type == 'sequential_bootstrapping_decision_tree':
                    clf = SequentiallyBootstrappedBaggingClassifier(samples_info_sets=events.loc[X_train.index].t1, ## events
                                                                    price_bars = dollar_bars.loc[X_train.index.min():X_train.index.max(), 'close'], ## df
                                                                    estimator=clf_base, 
                                                                    n_estimators=n_est, 
                                                                    random_state=RANDOM_STATE, 
                                                                    n_jobs=-1, 
                                                                    oob_score=False,
                                                                    max_features=1.)
                # get best param dict   
                best_param_dict = get_clf_best_param_cv(type, clf, X_train, y_train, cv_gen, scoring=scoring, sample_weight=sample_weight)

    return best_param_dict