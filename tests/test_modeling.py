"""Tests for baseline probability-modeling helpers."""

import unittest

import numpy as np
import pandas as pd

from src.features import build_features, extract_target
from src.modeling import (
    apply_platt_calibrator,
    calibration_table,
    constant_probability,
    evaluate_probabilities,
    fit_platt_calibrator,
    make_logistic_pipeline,
)


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = pd.DataFrame(
            {
                "SHOT_DISTANCE": [1, 3, 5, 10, 15, 20, 23, 24, 26, 30, 4, 8],
                "LOC_X": [0, 10, -20, 40, -80, 100, -220, 220, 40, -50, 5, -30],
                "LOC_Y": [10, 25, 45, 90, 120, 180, 60, 70, 250, 290, 35, 70],
                "PERIOD": [1, 1, 2, 2, 3, 3, 4, 4, 1, 5, 2, 4],
                "MINUTES_REMAINING": [11, 9, 8, 6, 4, 3, 0, 0, 10, 2, 5, 1],
                "SECONDS_REMAINING": [30, 20, 10, 5, 40, 15, 4, 20, 50, 30, 0, 45],
                "ACTION_TYPE": ["Layup", "Layup", "Hook", "Pullup", "Jump", "Jump", "Jump", "Jump", "Pullup", "Heave", "Layup", "Hook"],
                "SHOT_TYPE": ["2PT Field Goal"] * 6 + ["3PT Field Goal"] * 4 + ["2PT Field Goal"] * 2,
                "SHOT_ZONE_BASIC": ["Restricted Area"] * 3 + ["In The Paint (Non-RA)"] * 2 + ["Mid-Range"] + ["Left Corner 3", "Right Corner 3", "Above the Break 3", "Backcourt", "Restricted Area", "In The Paint (Non-RA)"],
                "SHOT_ZONE_AREA": ["Center(C)"] * 6 + ["Left Side(L)", "Right Side(R)", "Center(C)", "Back Court(BC)", "Center(C)", "Center(C)"],
                "SHOT_ZONE_RANGE": ["Less Than 8 ft."] * 3 + ["8-16 ft."] * 2 + ["16-24 ft."] + ["24+ ft."] * 4 + ["Less Than 8 ft.", "8-16 ft."],
                "PLAYER_ID": list(range(100, 112)),
                "TEAM_ID": [1] * 6 + [2] * 6,
                "SHOT_MADE_FLAG": [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            }
        )

    def test_constant_probability_uses_training_rate(self) -> None:
        predictions = constant_probability(pd.Series([1, 0, 1, 1]), row_count=3)
        np.testing.assert_allclose(predictions, [0.75, 0.75, 0.75])

    def test_probability_metrics_are_returned(self) -> None:
        metrics = evaluate_probabilities(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.8, 0.3, 0.7]),
        )

        self.assertEqual(set(metrics), {"log_loss", "brier_score", "ece"})
        self.assertGreater(metrics["log_loss"], 0)
        self.assertGreaterEqual(metrics["ece"], 0)

    def test_calibration_table_preserves_total_rows(self) -> None:
        table = calibration_table(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.8, 0.3, 0.7]),
            n_bins=5,
        )
        self.assertEqual(int(table["shots"].sum()), 4)

    def test_invalid_probabilities_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate_probabilities(np.array([0, 1]), np.array([0.2, 1.1]))

    def test_logistic_pipeline_fits_and_predicts(self) -> None:
        features = build_features(self.raw)
        target = extract_target(self.raw)
        model = make_logistic_pipeline(max_iter=500)
        model.fit(features, target)
        predictions = model.predict_proba(features)[:, 1]

        self.assertEqual(len(predictions), len(target))
        self.assertTrue(((predictions >= 0) & (predictions <= 1)).all())

    def test_platt_calibrator_returns_valid_probabilities(self) -> None:
        calibration_target = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        calibration_probabilities = np.array([0.05, 0.15, 0.30, 0.35, 0.45, 0.60, 0.75, 0.90])
        calibrator = fit_platt_calibrator(calibration_probabilities, calibration_target)
        calibrated = apply_platt_calibrator(calibrator, calibration_probabilities)

        self.assertEqual(len(calibrated), len(calibration_target))
        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())


if __name__ == "__main__":
    unittest.main()
