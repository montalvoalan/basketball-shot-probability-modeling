"""Locked final training and prediction workflow for the NBA shot model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.catboost_model import make_catboost_classifier
from src.diagnostics import history_aware_identity_predictions
from src.features import build_features, extract_target, feature_groups


FINAL_DEPTH = 5
FINAL_CONTEXT_ITERATIONS = 1_200
FINAL_IDENTITY_ITERATIONS = 612
MINIMUM_IDENTITY_HISTORY = 250


@dataclass(frozen=True)
class FinalModels:
    """The two fitted models required by the history-aware prediction rule."""

    context: CatBoostClassifier
    identity: CatBoostClassifier


def validate_shot_frame(shots: pd.DataFrame, *, require_target: bool) -> None:
    """Validate row identity and optional outcome requirements before modeling."""
    required_identifiers = ["GAME_ID", "GAME_EVENT_ID"]
    missing_identifiers = sorted(set(required_identifiers).difference(shots.columns))
    if missing_identifiers:
        raise ValueError(f"Missing event identifiers: {missing_identifiers}")
    if shots.duplicated(required_identifiers).any():
        raise ValueError("GAME_ID and GAME_EVENT_ID must uniquely identify every shot")
    if require_target:
        extract_target(shots)


def fit_final_models(
    development_shots: pd.DataFrame,
    *,
    depth: int = FINAL_DEPTH,
    context_iterations: int = FINAL_CONTEXT_ITERATIONS,
    identity_iterations: int = FINAL_IDENTITY_ITERATIONS,
    random_seed: int = 42,
) -> FinalModels:
    """Fit locked context and identity models on all development seasons."""
    validate_shot_frame(development_shots, require_target=True)
    target = extract_target(development_shots)

    context_features = build_features(development_shots)
    _, context_categorical = feature_groups(include_identity=False)
    context_model = make_catboost_classifier(
        depth=depth,
        iterations=context_iterations,
        random_seed=random_seed,
    )
    context_model.fit(
        context_features,
        target,
        cat_features=context_categorical,
        verbose=False,
    )

    identity_features = build_features(development_shots, include_identity=True)
    _, identity_categorical = feature_groups(include_identity=True)
    identity_model = make_catboost_classifier(
        depth=depth,
        iterations=identity_iterations,
        random_seed=random_seed,
    )
    identity_model.fit(
        identity_features,
        target,
        cat_features=identity_categorical,
        verbose=False,
    )

    return FinalModels(context=context_model, identity=identity_model)


def predict_final_probabilities(
    models: FinalModels,
    development_shots: pd.DataFrame,
    prediction_shots: pd.DataFrame,
    *,
    minimum_identity_history: int = MINIMUM_IDENTITY_HISTORY,
) -> pd.DataFrame:
    """Generate context, identity, and locked history-aware probabilities."""
    validate_shot_frame(development_shots, require_target=True)
    validate_shot_frame(prediction_shots, require_target=False)

    context_features = build_features(prediction_shots)
    identity_features = build_features(prediction_shots, include_identity=True)
    context_probabilities = models.context.predict_proba(context_features)[:, 1]
    identity_probabilities = models.identity.predict_proba(identity_features)[:, 1]
    history_aware_probabilities = history_aware_identity_predictions(
        context_probabilities,
        identity_probabilities,
        development_shots,
        prediction_shots,
        minimum_training_attempts=minimum_identity_history,
    )

    training_volume = development_shots["PLAYER_ID"].value_counts()
    player_attempts = prediction_shots["PLAYER_ID"].map(training_volume).fillna(0).astype(int)
    predictions = pd.DataFrame(
        {
            "context_probability": context_probabilities,
            "identity_probability": identity_probabilities,
            "history_aware_probability": history_aware_probabilities,
            "training_player_attempts": player_attempts.to_numpy(),
            "used_identity": (player_attempts >= minimum_identity_history).to_numpy(),
        },
        index=prediction_shots.index,
    )

    probability_columns = [
        "context_probability",
        "identity_probability",
        "history_aware_probability",
    ]
    if not predictions.index.equals(prediction_shots.index):
        raise AssertionError("Prediction row order changed")
    if len(predictions) != len(prediction_shots):
        raise AssertionError("Prediction row count changed")
    if not np.isfinite(predictions[probability_columns].to_numpy()).all():
        raise AssertionError("Predictions contain non-finite values")
    if not predictions[probability_columns].apply(lambda column: column.between(0, 1).all()).all():
        raise AssertionError("Predictions must be between 0 and 1")

    return predictions
