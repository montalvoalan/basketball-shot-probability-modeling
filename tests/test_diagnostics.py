"""Tests for basketball-focused segment diagnostics."""

import unittest

import numpy as np
import pandas as pd

from src.diagnostics import (
    distance_bands,
    history_aware_identity_predictions,
    period_clock_bands,
    player_volume_groups,
    segment_metrics,
)


class DiagnosticTests(unittest.TestCase):
    def test_basketball_segments_are_assigned(self) -> None:
        shots = pd.DataFrame(
            {
                "SHOT_DISTANCE": [2, 27, 55],
                "MINUTES_REMAINING": [0, 0, 8],
                "SECONDS_REMAINING": [4, 15, 30],
            }
        )

        self.assertEqual(distance_bands(shots).astype(str).tolist(), ["0-4", "25-29", "50+"])
        self.assertEqual(period_clock_bands(shots).astype(str).tolist(), ["0-5 sec", "6-24 sec", "6-12:00"])

    def test_player_volume_uses_training_only(self) -> None:
        training = pd.DataFrame({"PLAYER_ID": [1] * 800 + [2] * 100})
        validation = pd.DataFrame({"PLAYER_ID": [1, 2, 3]})

        groups = player_volume_groups(training, validation).astype(str).tolist()
        self.assertEqual(groups, ["750+", "1-249", "Unseen"])

    def test_segment_metrics_preserve_rows(self) -> None:
        target = np.array([0, 1, 0, 1])
        probabilities = np.array([0.1, 0.8, 0.4, 0.7])
        segments = pd.Series(["A", "A", "B", "B"])

        result = segment_metrics(target, probabilities, segments)
        self.assertEqual(int(result["shots"].sum()), 4)
        self.assertEqual(set(result["segment"]), {"A", "B"})

    def test_history_aware_identity_falls_back_for_limited_history(self) -> None:
        training = pd.DataFrame({"PLAYER_ID": [1] * 300 + [2] * 100})
        validation = pd.DataFrame({"PLAYER_ID": [1, 2, 3]})
        context = np.array([0.4, 0.5, 0.6])
        identity = np.array([0.7, 0.8, 0.9])

        combined = history_aware_identity_predictions(
            context, identity, training, validation, minimum_training_attempts=250
        )
        np.testing.assert_allclose(combined, [0.7, 0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
