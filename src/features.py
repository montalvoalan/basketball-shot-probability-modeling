"""Leakage-safe feature engineering for NBA shot probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "SHOT_MADE_FLAG"

CONTEXT_NUMERIC_FEATURES = (
    "shot_distance_feet",
    "shot_distance_squared",
    "loc_x_feet",
    "loc_y_feet",
    "abs_loc_x_feet",
    "shot_angle_degrees",
    "period",
    "seconds_remaining_in_period",
    "final_five_seconds",
    "is_overtime",
    "is_corner_three",
)

CONTEXT_CATEGORICAL_FEATURES = (
    "action_type",
    "shot_type",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
)

IDENTITY_CATEGORICAL_FEATURES = (
    "player_id",
    "team_id",
)

LEAKAGE_COLUMNS = (
    "SHOT_MADE_FLAG",
    "EVENT_TYPE",
)

_CONTEXT_SOURCE_COLUMNS = (
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
)

_IDENTITY_SOURCE_COLUMNS = (
    "PLAYER_ID",
    "TEAM_ID",
)


def feature_groups(include_identity: bool = False) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical columns produced by ``build_features``."""
    numeric = list(CONTEXT_NUMERIC_FEATURES)
    categorical = list(CONTEXT_CATEGORICAL_FEATURES)
    if include_identity:
        categorical.extend(IDENTITY_CATEGORICAL_FEATURES)
    return numeric, categorical


def _validate_source(shots: pd.DataFrame, include_identity: bool) -> None:
    """Fail early when the raw shot data does not satisfy the feature contract."""
    if not isinstance(shots, pd.DataFrame):
        raise TypeError("shots must be a pandas DataFrame")

    required = list(_CONTEXT_SOURCE_COLUMNS)
    if include_identity:
        required.extend(_IDENTITY_SOURCE_COLUMNS)

    missing_columns = sorted(set(required).difference(shots.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    columns_with_nulls = shots[required].columns[shots[required].isna().any()].tolist()
    if columns_with_nulls:
        raise ValueError(f"Required columns contain missing values: {columns_with_nulls}")

    if (shots["PERIOD"] < 1).any():
        raise ValueError("PERIOD must be at least 1")
    if not shots["MINUTES_REMAINING"].between(0, 12).all():
        raise ValueError("MINUTES_REMAINING must be between 0 and 12")
    if not shots["SECONDS_REMAINING"].between(0, 59).all():
        raise ValueError("SECONDS_REMAINING must be between 0 and 59")
    if (
        (shots["MINUTES_REMAINING"] == 12)
        & (shots["SECONDS_REMAINING"] != 0)
    ).any():
        raise ValueError("A 12-minute game clock must have zero remaining seconds")
    if (shots["SHOT_DISTANCE"] < 0).any():
        raise ValueError("SHOT_DISTANCE cannot be negative")


def build_features(shots: pd.DataFrame, include_identity: bool = False) -> pd.DataFrame:
    """Create model-ready shot features without using outcome information.

    The same function is applied independently to every temporal split. Player and
    team IDs are excluded by default and are available only for a controlled
    CatBoost ablation.
    """
    _validate_source(shots, include_identity=include_identity)

    features = pd.DataFrame(index=shots.index)

    loc_x_feet = shots["LOC_X"].astype(float) / 10.0
    loc_y_feet = shots["LOC_Y"].astype(float) / 10.0
    shot_distance = shots["SHOT_DISTANCE"].astype(float)
    seconds_remaining = (
        shots["MINUTES_REMAINING"].astype(int) * 60
        + shots["SECONDS_REMAINING"].astype(int)
    )

    features["shot_distance_feet"] = shot_distance
    features["shot_distance_squared"] = shot_distance**2
    features["loc_x_feet"] = loc_x_feet
    features["loc_y_feet"] = loc_y_feet
    features["abs_loc_x_feet"] = loc_x_feet.abs()
    features["shot_angle_degrees"] = np.degrees(
        np.arctan2(loc_x_feet.abs(), loc_y_feet)
    )
    features["period"] = shots["PERIOD"].astype(int)
    features["seconds_remaining_in_period"] = seconds_remaining
    features["final_five_seconds"] = (seconds_remaining <= 5).astype("int8")
    features["is_overtime"] = (shots["PERIOD"] > 4).astype("int8")
    features["is_corner_three"] = shots["SHOT_ZONE_BASIC"].isin(
        ["Left Corner 3", "Right Corner 3"]
    ).astype("int8")

    categorical_mapping = {
        "action_type": "ACTION_TYPE",
        "shot_type": "SHOT_TYPE",
        "shot_zone_basic": "SHOT_ZONE_BASIC",
        "shot_zone_area": "SHOT_ZONE_AREA",
        "shot_zone_range": "SHOT_ZONE_RANGE",
    }
    for feature_name, source_name in categorical_mapping.items():
        features[feature_name] = shots[source_name].astype(str)

    if include_identity:
        features["player_id"] = shots["PLAYER_ID"].astype(str)
        features["team_id"] = shots["TEAM_ID"].astype(str)

    numeric_features, categorical_features = feature_groups(include_identity)
    expected_columns = numeric_features + categorical_features

    if features.columns.tolist() != expected_columns:
        raise AssertionError("Feature columns do not match the declared schema")
    if set(features.columns).intersection(LEAKAGE_COLUMNS):
        raise AssertionError("Target-derived columns entered the feature set")
    if features.isna().any().any():
        raise AssertionError("Feature engineering created missing values")

    return features


def extract_target(shots: pd.DataFrame) -> pd.Series:
    """Validate and return the binary shot outcome with its original index."""
    if TARGET_COLUMN not in shots.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if shots[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values")

    target_values = set(shots[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary; found {sorted(target_values)}")

    return shots[TARGET_COLUMN].astype("int8").rename("shot_made")
