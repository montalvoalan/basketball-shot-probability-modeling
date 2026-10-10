"""Compact CatBoost probability model for NBA shot outcomes."""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from src.features import feature_groups


def make_catboost_classifier(
    *,
    depth: int,
    iterations: int = 1_200,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
) -> CatBoostClassifier:
    """Create a reproducible CatBoost classifier with restrained complexity."""
    if not 2 <= depth <= 10:
        raise ValueError("depth must be between 2 and 10")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if l2_leaf_reg < 0:
        raise ValueError("l2_leaf_reg cannot be negative")

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )


def fit_catboost_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    include_identity: bool = False,
    depth: int,
    iterations: int = 1_200,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 5.0,
    early_stopping_rounds: int = 60,
    random_seed: int = 42,
) -> CatBoostClassifier:
    """Fit CatBoost using one temporal validation fold for early stopping."""
    numeric_features, categorical_features = feature_groups(include_identity)
    expected_columns = numeric_features + categorical_features

    if x_train.columns.tolist() != expected_columns:
        raise ValueError("Training features do not match the declared feature schema")
    if x_validation.columns.tolist() != expected_columns:
        raise ValueError("Validation features do not match the declared feature schema")
    if x_train.isna().any().any() or x_validation.isna().any().any():
        raise ValueError("CatBoost inputs cannot contain missing values")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ValueError("Feature and target row counts must match")
    if early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive")

    model = make_catboost_classifier(
        depth=depth,
        iterations=iterations,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(x_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model
