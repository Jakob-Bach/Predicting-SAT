"""Prediction

Functions for making and evaluating predictions.
"""


import pandas as pd
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
NUM_FEATURES_LIST = [5, 10, 'all']

MODELS = [
    {'name': 'Decision tree', 'func': sklearn.tree.DecisionTreeClassifier,
     'args': {'random_state': 25}},
    {'name': 'kNN', 'func': sklearn.neighbors.KNeighborsClassifier,
     'args': {'n_jobs': 1}},
    {'name': 'Random forest', 'func': sklearn.ensemble.RandomForestClassifier,
     'args': {'n_estimators': 100, 'random_state': 25, 'n_jobs': 1}},
    {'name': 'XGBoost', 'func': xgboost.XGBClassifier,
     'args': {'n_estimators': 100, 'random_state': 25, 'n_jobs': 1,
              'booster': 'gbtree', 'objective': 'binary:logistic',  # also handles multi-class
              'use_label_encoder': False, 'verbosity': 0}}
]

METRICS = {'acc': sklearn.metrics.accuracy_score, 'mcc': sklearn.metrics.matthews_corrcoef}


# Evaluate all prediction models for all feature-selection settings on the cross-validation fold
# identified by "fold_id" on the dataset given by "X" and "y".
# Return a table with train/test performance regarding multiple evaluation metrics.
def predict_and_evaluate(X: pd.DataFrame, y: pd.Series, fold_id: int) -> pd.DataFrame:
    imputer = sklearn.impute.SimpleImputer(strategy='mean')
    label_encoder = sklearn.preprocessing.LabelEncoder()
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
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    y_train_encoded = label_encoder.fit_transform(y=y_train)
    filter_fs_scores = pd.Series(sklearn.feature_selection.mutual_info_classif(
        X=X_train, y=y_train, discrete_features=False, random_state=25), index=X_train.columns)
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
            model.fit(X=X_train_selected, y=y_train_encoded)
            pred_train = label_encoder.inverse_transform(model.predict(X_train_selected))
            pred_test = label_encoder.inverse_transform(model.predict(X_test_selected))
            result = {'fold_id': fold_id, 'model_name': model_item['name'],
                      'num_features': num_features}
            for metric_name, metric_func in METRICS.items():
                result[f'train_{metric_name}'] = metric_func(y_true=y_train, y_pred=pred_train)
                result[f'test_{metric_name}'] = metric_func(y_true=y_test, y_pred=pred_test)
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            else:
                feature_importances = [float('nan')] * len(X_train_selected.columns)
            result.update({f'imp.{feature_name}': importance for (feature_name, importance)
                           in zip(X_train_selected.columns, feature_importances)})
            results.append(result)
    return pd.DataFrame(results)
