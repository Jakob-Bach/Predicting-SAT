"""Prediction

Functions for making and evaluating predictions.
"""


import warnings

import pandas as pd
import shap
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.impute
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.tree
import xgboost


NUM_CV_FOLDS = 10
NUM_FEATURES_LIST = [1, 2, 3, 4, 5, 10, 20, 'all']

MODELS = [
    {'name': 'Decision tree', 'func': sklearn.tree.DecisionTreeClassifier,
     'args': {'random_state': 25}},
    {'name': 'kNN', 'func': sklearn.neighbors.KNeighborsClassifier,
     'args': {'n_jobs': 1}},
    {'name': 'Random forest', 'func': sklearn.ensemble.RandomForestClassifier,
     'args': {'n_estimators': 100, 'random_state': 25, 'n_jobs': 1}},
    {'name': 'XGBoost', 'func': xgboost.XGBClassifier,
     'args': {'n_estimators': 100, 'random_state': 25, 'n_jobs': 1, 'booster': 'gbtree',
              'objective': 'binary:logistic', 'verbosity': 0}}  # also supports multi-class
]

METRICS = {'acc': sklearn.metrics.accuracy_score, 'mcc': sklearn.metrics.matthews_corrcoef}

TARGET_ENCODING = {'unsat': 0, 'sat': 1, 'unknown': 2}  # manual, to fix the label of "sat" for SHAP


# Evaluate all prediction models for all feature-selection settings on the cross-validation fold
# identified by "fold_id" on the dataset given by the feature part "X" and the target "y".
# "families" is used for evaluation (of misclassifications) only.
# Return a table with train/test performance regarding multiple evaluation metrics.
def predict_and_evaluate(X: pd.DataFrame, y: pd.Series, families: pd.Series, fold_id: int) -> pd.DataFrame:
    imputer = sklearn.impute.SimpleImputer(strategy='mean')
    scaler = sklearn.preprocessing.MinMaxScaler()
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True,
                                                       random_state=25)
    train_idx, test_idx = list(splitter.split(X=X, y=y))[fold_id]
    results = []
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    X_train = pd.DataFrame(imputer.fit_transform(X=X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X=X_test), columns=X_test.columns)
    X_train = pd.DataFrame(scaler.fit_transform(X=X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X=X_test), columns=X_test.columns)
    y_train = y.iloc[train_idx].reset_index(drop=True).replace(TARGET_ENCODING)
    y_test = y.iloc[test_idx].reset_index(drop=True).replace(TARGET_ENCODING)
    families_test = families.iloc[test_idx].reset_index(drop=True)  # not used for predictions
    filter_fs_scores = pd.Series(sklearn.feature_selection.mutual_info_classif(
        X=X_train, y=y_train, discrete_features=False, random_state=25), index=X_train.columns)
    filter_fs_scores = filter_fs_scores / filter_fs_scores.sum()  # normalize (as model importances)
    filter_fs_scores = filter_fs_scores.sort_values(ascending=False)  # avoid sorting for each k
    for num_features in NUM_FEATURES_LIST:
        if num_features == 'all':
            selected_features = X_train.columns
        else:
            selected_features = filter_fs_scores.index[:num_features]  # sorted beforehand!
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        for model_item in MODELS:
            model = model_item['func'](**model_item['args'])
            model.fit(X=X_train_selected, y=y_train)
            pred_train = model.predict(X_train_selected)
            pred_test = model.predict(X_test_selected)
            result = {'fold_id': fold_id, 'model_name': model_item['name'],
                      'num_features': num_features}
            for metric_name, metric_func in METRICS.items():
                result[f'train_{metric_name}'] = metric_func(y_true=y_train, y_pred=pred_train)
                result[f'test_{metric_name}'] = metric_func(y_true=y_test, y_pred=pred_test)
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            else:
                feature_importances = [float('nan')] * len(X_train_selected.columns)
            result.update({f'imp_fs.{feature_name}': importance for (feature_name, importance)
                           in filter_fs_scores.iteritems()})
            result.update({f'imp_mod.{feature_name}': importance for (feature_name, importance)
                           in zip(X_train_selected.columns, feature_importances)})
            if model_item['name'] != 'kNN':
                shap_explainer = shap.TreeExplainer(model=model)
                # Compute SHAP values for each train instance and class 1 (SAT), make absolute (can
                # be negative), normalize (sum to 1 per instance), average over instances:
                with warnings.catch_warnings():  # shap uses deprecated xgboost parameter
                    warnings.filterwarnings(action='ignore', message='ntree_limit is deprecated')
                    feature_importances = shap_explainer.shap_values(X=X_train_selected)[1]
                feature_importances = pd.DataFrame(feature_importances).abs()
                feature_importances = feature_importances.div(
                    feature_importances.sum(axis='columns'), axis='rows')
                feature_importances = feature_importances.mean()
                result.update({f'imp_shap.{feature_name}': importance for (feature_name, importance)
                               in zip(X_train_selected.columns, feature_importances)})
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='An input array is constant')
                result.update({f'miscl_corr.{feature_name}': miscl_corr for (feature_name, miscl_corr)
                               in X_test.corrwith(y_test != pred_test, method='spearman').iteritems()})
            result.update({f'miscl_freq.{family_name}': miscl_freq for (family_name, miscl_freq)
                           in pd.DataFrame({'family': families_test, 'misc': y_test != pred_test}
                                           ).groupby('family')['misc'].mean().iteritems()})
            results.append(result)
    return pd.DataFrame(results)
