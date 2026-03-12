# -*- coding: utf-8 -*-
"""
Machine learning classification module.

Supports 6 classifiers with 5-fold cross-validation, optional data
augmentation (tSMOTE / tsaug), and StandardScaler preprocessing.
"""

import time

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from augmentation import apply_tsmote, apply_tsaug


ML_METHODS = [
    ("KNeighborsClassifier",
     lambda: KNeighborsClassifier(
         n_neighbors=5, weights="uniform", metric="minkowski", leaf_size=30)),
    ("DecisionTreeClassifier",
     lambda: DecisionTreeClassifier(
         criterion="gini", random_state=42, splitter="best", max_depth=None)),
    ("RandomForestClassifier",
     lambda: RandomForestClassifier(
         n_estimators=100, criterion="gini", random_state=42,
         max_features="sqrt")),
    ("XGBoost",
     lambda: xgb.XGBClassifier(
         objective="multi:softmax", num_class=2, random_state=42)),
    ("LightGBM",
     lambda: lgb.LGBMClassifier(
         boosting_type="gbdt", num_leaves=15, max_depth=-1,
         learning_rate=0.1, n_estimators=100, random_state=42)),
    ("SVC",
     lambda: SVC(C=1.0, kernel="rbf", degree=3)),
]


def classify(all_data, all_labels, classifier_index,
             use_tsmote=False, use_tsaug=False):
    """Train and evaluate a classifier using 5-fold cross-validation.

    Parameters
    ----------
    all_data : np.ndarray
    all_labels : np.ndarray
    classifier_index : int
        Index into ML_METHODS.
    use_tsmote : bool
        Apply tSMOTE augmentation on train folds.
    use_tsaug : bool
        Apply tsaug augmentation on train folds.

    Returns
    -------
    results : str
        Formatted string with averaged metrics.
    """
    scaler = StandardScaler()

    try:
        model = ML_METHODS[classifier_index][1]()
    except IndexError:
        print("Invalid classifier id!")
        return ""

    if use_tsmote or use_tsaug:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0
    avg_accuracy = 0.0
    avg_precision = 0.0
    avg_recall = 0.0
    avg_f1 = 0.0
    avg_train_time = 0.0
    avg_test_time = 0.0

    for train_index, test_index in kf.split(all_data, all_labels):
        fold += 1

        X_train = scaler.fit_transform(all_data[train_index, :])
        X_test = scaler.transform(all_data[test_index, :])
        y_train = all_labels[train_index]
        y_test = all_labels[test_index]

        training_start = time.perf_counter()

        if use_tsmote:
            X_train, y_train = apply_tsmote(
                X_train, y_train,
                nSlices=2, nPoints=5, nFix=0, oversample_ratio=10,
            )
        elif use_tsaug:
            X_train, y_train = apply_tsaug(X_train, y_train, aug_ratio=10)

        model.fit(X_train, y_train)
        training_time = time.perf_counter() - training_start
        avg_train_time += training_time

        test_start = time.perf_counter()
        y_pred = model.predict(X_test)
        test_time = time.perf_counter() - test_start
        avg_test_time += test_time

        avg_accuracy += accuracy_score(y_test, y_pred)
        avg_precision += precision_score(y_test, y_pred)
        avg_recall += recall_score(y_test, y_pred)
        avg_f1 += f1_score(y_test, y_pred)

    avg_accuracy /= fold
    avg_precision /= fold
    avg_recall /= fold
    avg_f1 /= fold
    avg_train_time /= fold
    avg_test_time /= fold

    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1-Score: {avg_f1}")
    print(f"Average training time: {avg_train_time}")
    print(f"Average test time: {avg_test_time}")

    results = (
        f"Model: {model}\n"
        f"Avg. Accuracy: {avg_accuracy}\n"
        f"Avg. Precision: {avg_precision}\n"
        f"Avg. Recall: {avg_recall}\n"
        f"Avg. F1: {avg_f1}\n"
        f"Avg. Training Time: {avg_train_time}\n"
        f"Avg. Test Time: {avg_test_time}\n"
    )
    return results
