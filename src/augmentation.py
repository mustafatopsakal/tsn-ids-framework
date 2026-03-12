# -*- coding: utf-8 -*-
"""
Data augmentation strategies for handling class imbalance in
time-series classification.

Supports two methods:
- tSMOTE: time-series SMOTE via slicing, interpolation, and imputation
- tsaug:  time-series augmentation with TimeWarp, Drift, and AddNoise
"""

import numpy as np
from collections import Counter

import tsmote
from tsaug import TimeWarp, Drift, AddNoise


# ---------------------------------------------------------------------------
# tSMOTE augmentation
# ---------------------------------------------------------------------------

def _apply_tsmote_single(sample, tMin, tMax, nSlices, nPoints, nFix,
                         oversample_ratio):
    """Apply tSMOTE oversampling to a single sample (time series).

    If the sample has only one observation, small noise is added instead.
    """
    synthetic_samples = []

    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    sample = sample.astype(float)
    sample_list = sample.tolist()

    time_stamps = [row[0] for row in sample_list]

    if len(time_stamps) < 2:
        for _ in range(oversample_ratio):
            noise = np.random.normal(0, 0.001, sample.shape)
            synthetic_samples.append(sample.copy() + noise)
        return np.array(synthetic_samples)

    T_input = [time_stamps]
    bins, sliceLen = tsmote.getNonUniformTimeSliceBins(
        T_input, tMin, tMax, nSlices
    )

    tSlices = tsmote.getRawTimeSlice(sample_list, bins, nSlices)
    tSliceSyn = tsmote.generateTimePoints(tSlices, nPoints)
    imputed = tsmote.imputeTimeSlices(sample_list, bins, tSliceSyn, nFix)

    for _ in range(oversample_ratio):
        synthetic_samples.append(np.array(imputed[0], dtype=float))
    return np.array(synthetic_samples)


def apply_tsmote(X, y, nSlices=2, nPoints=5, nFix=0, oversample_ratio=10):
    """Apply tSMOTE oversampling on a per-class basis.

    Original minority samples are preserved; synthetic samples are added.

    Returns
    -------
    X_aug_flat : np.ndarray (2D)
    y_aug : np.ndarray (1D)
    """
    classes = np.unique(y)
    X_list = []
    y_list = []

    for c in classes:
        indices = np.where(y == c)[0]
        X_c = X[indices].astype(float)

        if c == 1:
            if X_c.ndim == 2:
                X_c = X_c.reshape((-1, 1, X_c.shape[1]))

            X_c_aug = list(X_c)

            all_time_stamps = []
            for sample in X_c:
                sample_list = sample.tolist()
                all_time_stamps.extend([row[0] for row in sample_list])
            common_tMin = min(all_time_stamps)
            common_tMax = max(all_time_stamps)

            for sample in X_c:
                synthetic = _apply_tsmote_single(
                    sample, common_tMin, common_tMax,
                    nSlices, nPoints, nFix, oversample_ratio,
                )
                for s in synthetic:
                    X_c_aug.append(s)
            X_c_aug = np.array(X_c_aug)
        else:
            if X_c.ndim == 2:
                X_c_aug = X_c.reshape((-1, 1, X_c.shape[1]))
            else:
                X_c_aug = X_c

        X_list.append(X_c_aug)
        y_list.append(np.full(X_c_aug.shape[0], c))

    X_aug = np.vstack(X_list)
    y_aug = np.concatenate(y_list)
    X_aug_flat = np.array([s.flatten() for s in X_aug])
    return X_aug_flat, y_aug


# ---------------------------------------------------------------------------
# tsaug augmentation
# ---------------------------------------------------------------------------

def apply_tsaug(X, y, aug_ratio=10):
    """Augment minority-class time series using tsaug transforms.

    Applies TimeWarp -> Drift -> AddNoise sequentially.
    Original data is preserved and synthetic samples are appended.

    Returns
    -------
    X_augmented : np.ndarray (2D)
    y_augmented : np.ndarray (1D)
    """
    X_synthetic_all = []
    y_synthetic_all = []

    majority_count = max(Counter(y).values())

    for class_label in set(y):
        indices = y == class_label
        X_class = X[indices]

        if X_class.shape[0] < majority_count:
            synthetic_samples = []
            for sample in X_class:
                sample_reshaped = sample.reshape(1, -1)
                for _ in range(aug_ratio):
                    augmented = TimeWarp(
                        n_speed_change=3, max_speed_ratio=2
                    ).augment(sample_reshaped)
                    augmented = Drift(
                        max_drift=0.1, n_drift_points=5
                    ).augment(augmented)
                    augmented = AddNoise(scale=0.001).augment(augmented)
                    synthetic_samples.append(augmented.squeeze())

            synthetic_array = np.vstack(synthetic_samples)
            X_synthetic_all.append(synthetic_array)
            y_synthetic_all.append(
                np.full(synthetic_array.shape[0], class_label)
            )

    if X_synthetic_all:
        X_synthetic_all = np.vstack(X_synthetic_all)
        y_synthetic_all = np.hstack(y_synthetic_all)
        X_augmented = np.vstack([X, X_synthetic_all])
        y_augmented = np.hstack([y, y_synthetic_all])
    else:
        X_augmented, y_augmented = X, y

    return X_augmented, y_augmented
