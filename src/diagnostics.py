"""Basketball-focused segment diagnostics for shot probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def distance_bands(shots: pd.DataFrame) -> pd.Series:
    """Return interpretable shot-distance bands in feet."""
    return pd.cut(
        shots["SHOT_DISTANCE"],
        bins=[-1, 4, 9, 14, 19, 24, 29, 39, 49, np.inf],
        labels=["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-39", "40-49", "50+"],
    ).rename("distance_band")


def period_clock_bands(shots: pd.DataFrame) -> pd.Series:
    """Return game-clock bands without implying possession shot-clock data."""
    seconds_remaining = shots["MINUTES_REMAINING"] * 60 + shots["SECONDS_REMAINING"]
    return pd.cut(
        seconds_remaining,
        bins=[-1, 5, 24, 119, 359, 720],
        labels=["0-5 sec", "6-24 sec", "25-119 sec", "2-5:59", "6-12:00"],
    ).rename("period_clock_band")


def player_volume_groups(
    training_shots: pd.DataFrame,
    validation_shots: pd.DataFrame,
) -> pd.Series:
    """Group validation attempts by the player's training-fold shot volume."""
    training_volume = training_shots["PLAYER_ID"].value_counts()
    validation_volume = validation_shots["PLAYER_ID"].map(training_volume).fillna(0)
    return pd.cut(
        validation_volume,
        bins=[-1, 0, 249, 749, np.inf],
        labels=["Unseen", "1-249", "250-749", "750+"],
    ).rename("player_training_volume")


def history_aware_identity_predictions(
    context_probabilities: np.ndarray,
    identity_probabilities: np.ndarray,
    training_shots: pd.DataFrame,
    validation_shots: pd.DataFrame,
    *,
    minimum_training_attempts: int = 250,
) -> np.ndarray:
    """Use identity only when a player has enough prior training attempts."""
    context_array = np.asarray(context_probabilities, dtype=float).reshape(-1)
    identity_array = np.asarray(identity_probabilities, dtype=float).reshape(-1)
    if len(context_array) != len(identity_array) or len(context_array) != len(validation_shots):
        raise ValueError("Prediction and validation row counts must match")
    if minimum_training_attempts <= 0:
        raise ValueError("minimum_training_attempts must be positive")
    if not np.isfinite(context_array).all() or not np.isfinite(identity_array).all():
        raise ValueError("Probabilities must be finite")
    if ((context_array < 0) | (context_array > 1)).any() or ((identity_array < 0) | (identity_array > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    training_volume = training_shots["PLAYER_ID"].value_counts()
    validation_volume = validation_shots["PLAYER_ID"].map(training_volume).fillna(0)
    use_identity = validation_volume.to_numpy() >= minimum_training_attempts
    return np.where(use_identity, identity_array, context_array)


def segment_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    segments: pd.Series,
    *,
    minimum_shots: int = 1,
) -> pd.DataFrame:
    """Evaluate log loss, Brier score, and calibration gap within segments."""
    y_array = np.asarray(y_true).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=float).reshape(-1)
    segment_array = pd.Series(segments).reset_index(drop=True)

    if not (len(y_array) == len(probability_array) == len(segment_array)):
        raise ValueError("Target, probability, and segment lengths must match")
    if minimum_shots <= 0:
        raise ValueError("minimum_shots must be positive")
    if segment_array.isna().any():
        raise ValueError("Segments cannot contain missing values")
    if not set(np.unique(y_array)).issubset({0, 1}):
        raise ValueError("Target must be binary")
    if not np.isfinite(probability_array).all() or ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")

    evaluation = pd.DataFrame(
        {
            "segment": segment_array,
            "actual": y_array.astype(int),
            "predicted": probability_array,
        }
    )
    clipped = np.clip(evaluation["predicted"], 1e-15, 1 - 1e-15)
    evaluation["log_loss_row"] = -(
        evaluation["actual"] * np.log(clipped)
        + (1 - evaluation["actual"]) * np.log(1 - clipped)
    )
    evaluation["brier_row"] = (evaluation["actual"] - evaluation["predicted"]) ** 2

    summary = (
        evaluation.groupby("segment", observed=True)
        .agg(
            shots=("actual", "size"),
            observed_make_rate=("actual", "mean"),
            average_prediction=("predicted", "mean"),
            log_loss=("log_loss_row", "mean"),
            brier_score=("brier_row", "mean"),
        )
        .reset_index()
    )
    summary["calibration_gap"] = summary["average_prediction"] - summary["observed_make_rate"]
    summary["absolute_calibration_gap"] = summary["calibration_gap"].abs()
    return summary.loc[summary["shots"] >= minimum_shots].reset_index(drop=True)
