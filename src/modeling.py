"""Probability-modeling helpers shared by the baseline and later model phases."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import feature_groups


def constant_probability(y_train: pd.Series, row_count: int) -> np.ndarray:
    """Predict the training-fold make rate for every validation row."""
    if row_count < 0:
        raise ValueError("row_count cannot be negative")
    if len(y_train) == 0:
        raise ValueError("y_train cannot be empty")
    if y_train.isna().any() or not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("y_train must be a complete binary target")

    return np.full(row_count, float(y_train.mean()))


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error (lower is better)."""
    y_array, probability_array = _validated_arrays(y_true, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probability_array, bin_edges[1:-1], right=True)

    calibration_error = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if not in_bin.any():
            continue

        bin_weight = in_bin.mean()
        average_prediction = probability_array[in_bin].mean()
        observed_rate = y_array[in_bin].mean()
        calibration_error += bin_weight * abs(average_prediction - observed_rate)

    return float(calibration_error)


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Evaluate probability accuracy and calibration with complementary metrics."""
    y_array, probability_array = _validated_arrays(y_true, probabilities)

    return {
        "log_loss": float(log_loss(y_array, probability_array, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_array, probability_array)),
        "ece": expected_calibration_error(y_array, probability_array, n_bins=n_bins),
    }


def calibration_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed make rates in equal-width bins."""
    y_array, probability_array = _validated_arrays(y_true, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    calibration = pd.DataFrame(
        {
            "actual": y_array,
            "predicted": probability_array,
            "probability_bin": pd.cut(
                probability_array,
                bins=np.linspace(0.0, 1.0, n_bins + 1),
                include_lowest=True,
            ),
        }
    )

    return (
        calibration.groupby("probability_bin", observed=False)
        .agg(
            shots=("actual", "size"),
            average_prediction=("predicted", "mean"),
            observed_make_rate=("actual", "mean"),
        )
        .reset_index()
    )


def fit_platt_calibrator(
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
) -> LogisticRegression:
    """Fit a one-variable logistic calibration model on out-of-time predictions."""
    y_array, probability_array = _validated_arrays(y_true, probabilities)
    logits = _probability_logits(probability_array).reshape(-1, 1)

    calibrator = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1_000)
    calibrator.fit(logits, y_array)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Apply a fitted Platt calibrator to uncalibrated probabilities."""
    probability_array = np.asarray(probabilities, dtype=float).reshape(-1)
    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must be finite")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    logits = _probability_logits(probability_array).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def make_logistic_pipeline(
    *,
    include_identity: bool = False,
    regularization_strength: float = 1.0,
    max_iter: int = 1_000,
) -> Pipeline:
    """Build an L2-regularized logistic pipeline with fold-local preprocessing."""
    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive")

    numeric_features, categorical_features = feature_groups(include_identity)
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=regularization_strength,
                    solver="lbfgs",
                    max_iter=max_iter,
                ),
            ),
        ]
    )


def _validated_arrays(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate metric inputs and return one-dimensional NumPy arrays."""
    y_array = np.asarray(y_true).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(y_array) == 0:
        raise ValueError("Evaluation data cannot be empty")
    if len(y_array) != len(probability_array):
        raise ValueError("Target and probability lengths do not match")
    if not set(np.unique(y_array)).issubset({0, 1}):
        raise ValueError("Target must be binary")
    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must be finite")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    return y_array.astype(int), probability_array


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite logits for calibration."""
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))
