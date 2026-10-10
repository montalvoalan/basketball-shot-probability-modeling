"""Tests for the compact CatBoost model helper."""

import unittest

import numpy as np
import pandas as pd

from src.catboost_model import fit_catboost_classifier, make_catboost_classifier
from src.features import build_features, extract_target


class CatBoostModelTests(unittest.TestCase):
    @staticmethod
    def _sample_shots(row_count: int = 60) -> pd.DataFrame:
        rows = []
        action_types = ["Layup", "Jump Shot", "Pullup", "Dunk"]
        for row_id in range(row_count):
            distance = row_id % 30
            is_three = distance >= 23
            made = int((distance < 6 and row_id % 4 != 0) or (6 <= distance < 23 and row_id % 3 == 0) or (is_three and row_id % 4 == 0))
            rows.append(
                {
                    "SHOT_DISTANCE": distance,
                    "LOC_X": (row_id % 21 - 10) * 10,
                    "LOC_Y": distance * 10,
                    "PERIOD": row_id % 5 + 1,
                    "MINUTES_REMAINING": row_id % 12,
                    "SECONDS_REMAINING": row_id % 60,
                    "ACTION_TYPE": action_types[row_id % len(action_types)],
                    "SHOT_TYPE": "3PT Field Goal" if is_three else "2PT Field Goal",
                    "SHOT_ZONE_BASIC": "Above the Break 3" if is_three else "Restricted Area" if distance < 5 else "Mid-Range",
                    "SHOT_ZONE_AREA": "Center(C)",
                    "SHOT_ZONE_RANGE": "24+ ft." if is_three else "Less Than 8 ft." if distance < 8 else "8-16 ft." if distance < 16 else "16-24 ft.",
                    "PLAYER_ID": 100 + row_id % 8,
                    "TEAM_ID": 1 + row_id % 3,
                    "SHOT_MADE_FLAG": made,
                }
            )
        return pd.DataFrame(rows)

    def test_invalid_depth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth"):
            make_catboost_classifier(depth=1)

    def test_model_fits_and_returns_probabilities(self) -> None:
        raw = self._sample_shots()
        train_raw = raw.iloc[:45].reset_index(drop=True)
        validation_raw = raw.iloc[45:].reset_index(drop=True)

        model = fit_catboost_classifier(
            build_features(train_raw),
            extract_target(train_raw),
            build_features(validation_raw),
            extract_target(validation_raw),
            depth=3,
            iterations=40,
            early_stopping_rounds=8,
        )
        probabilities = model.predict_proba(build_features(validation_raw))[:, 1]

        self.assertEqual(len(probabilities), len(validation_raw))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertGreaterEqual(model.get_best_iteration(), 0)


if __name__ == "__main__":
    unittest.main()
