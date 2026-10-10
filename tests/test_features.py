"""Tests for the reusable basketball shot feature contract."""

import unittest

import pandas as pd

from src.features import (
    CONTEXT_CATEGORICAL_FEATURES,
    CONTEXT_NUMERIC_FEATURES,
    IDENTITY_CATEGORICAL_FEATURES,
    build_features,
    extract_target,
)


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shots = pd.DataFrame(
            {
                "SHOT_DISTANCE": [3, 24, 18],
                "LOC_X": [10, -220, 80],
                "LOC_Y": [25, 60, 160],
                "PERIOD": [1, 4, 5],
                "MINUTES_REMAINING": [11, 0, 2],
                "SECONDS_REMAINING": [17, 4, 30],
                "ACTION_TYPE": ["Driving Layup Shot", "Jump Shot", "Pullup Jump shot"],
                "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal", "2PT Field Goal"],
                "SHOT_ZONE_BASIC": ["Restricted Area", "Left Corner 3", "Mid-Range"],
                "SHOT_ZONE_AREA": ["Center(C)", "Left Side(L)", "Right Side(R)"],
                "SHOT_ZONE_RANGE": ["Less Than 8 ft.", "24+ ft.", "16-24 ft."],
                "PLAYER_ID": [101, 202, 303],
                "TEAM_ID": [10, 20, 30],
                "SHOT_MADE_FLAG": [1, 0, 1],
                "EVENT_TYPE": ["Made Shot", "Missed Shot", "Made Shot"],
            },
            index=[9, 4, 12],
        )

    def test_context_schema_and_row_order(self) -> None:
        features = build_features(self.shots)
        expected = list(CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES)

        self.assertEqual(features.columns.tolist(), expected)
        self.assertEqual(features.index.tolist(), [9, 4, 12])
        self.assertNotIn("SHOT_MADE_FLAG", features.columns)
        self.assertNotIn("EVENT_TYPE", features.columns)

    def test_basketball_features_are_calculated_correctly(self) -> None:
        features = build_features(self.shots)

        self.assertEqual(features.loc[4, "seconds_remaining_in_period"], 4)
        self.assertEqual(features.loc[4, "final_five_seconds"], 1)
        self.assertEqual(features.loc[4, "is_corner_three"], 1)
        self.assertEqual(features.loc[12, "is_overtime"], 1)
        self.assertAlmostEqual(features.loc[4, "loc_x_feet"], -22.0)
        self.assertAlmostEqual(features.loc[12, "shot_distance_squared"], 324.0)

    def test_identity_is_optional(self) -> None:
        context_only = build_features(self.shots)
        with_identity = build_features(self.shots, include_identity=True)

        self.assertTrue(set(IDENTITY_CATEGORICAL_FEATURES).isdisjoint(context_only.columns))
        self.assertTrue(set(IDENTITY_CATEGORICAL_FEATURES).issubset(with_identity.columns))

    def test_schema_matches_across_splits(self) -> None:
        first_split = build_features(self.shots.iloc[:2])
        second_split = build_features(self.shots.iloc[2:])

        self.assertEqual(first_split.columns.tolist(), second_split.columns.tolist())
        self.assertEqual(first_split.dtypes.astype(str).tolist(), second_split.dtypes.astype(str).tolist())

    def test_source_data_is_not_modified(self) -> None:
        original = self.shots.copy(deep=True)
        build_features(self.shots)
        pd.testing.assert_frame_equal(self.shots, original)

    def test_target_is_validated_and_preserves_index(self) -> None:
        target = extract_target(self.shots)

        self.assertEqual(target.tolist(), [1, 0, 1])
        self.assertEqual(target.index.tolist(), [9, 4, 12])
        self.assertEqual(target.name, "shot_made")

    def test_invalid_clock_is_rejected(self) -> None:
        invalid = self.shots.copy()
        invalid.loc[9, "SECONDS_REMAINING"] = 60

        with self.assertRaisesRegex(ValueError, "SECONDS_REMAINING"):
            build_features(invalid)


if __name__ == "__main__":
    unittest.main()
