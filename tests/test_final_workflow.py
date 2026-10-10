"""Tests for the locked final training and prediction workflow."""

import unittest

import pandas as pd

from src.final_workflow import (
    fit_final_models,
    predict_final_probabilities,
    validate_shot_frame,
)


class FinalWorkflowTests(unittest.TestCase):
    @staticmethod
    def _sample_shots(row_count: int, game_id: int, *, include_target: bool = True) -> pd.DataFrame:
        rows = []
        for row_id in range(row_count):
            distance = row_id % 30
            row = {
                "GAME_ID": game_id,
                "GAME_EVENT_ID": row_id,
                "SHOT_DISTANCE": distance,
                "LOC_X": (row_id % 21 - 10) * 10,
                "LOC_Y": distance * 10,
                "PERIOD": row_id % 5 + 1,
                "MINUTES_REMAINING": row_id % 12,
                "SECONDS_REMAINING": row_id % 60,
                "ACTION_TYPE": ["Layup", "Jump Shot", "Pullup", "Dunk"][row_id % 4],
                "SHOT_TYPE": "3PT Field Goal" if distance >= 23 else "2PT Field Goal",
                "SHOT_ZONE_BASIC": "Above the Break 3" if distance >= 23 else "Restricted Area" if distance < 5 else "Mid-Range",
                "SHOT_ZONE_AREA": "Center(C)",
                "SHOT_ZONE_RANGE": "24+ ft." if distance >= 23 else "Less Than 8 ft." if distance < 8 else "8-16 ft." if distance < 16 else "16-24 ft.",
                "PLAYER_ID": 100 + row_id % 8,
                "TEAM_ID": 1 + row_id % 3,
            }
            if include_target:
                row["SHOT_MADE_FLAG"] = int((distance < 6 and row_id % 4 != 0) or row_id % 5 == 0)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_duplicate_event_keys_are_rejected(self) -> None:
        shots = self._sample_shots(10, game_id=1)
        shots.loc[1, "GAME_EVENT_ID"] = shots.loc[0, "GAME_EVENT_ID"]
        with self.assertRaisesRegex(ValueError, "uniquely identify"):
            validate_shot_frame(shots, require_target=True)

    def test_final_models_preserve_prediction_rows(self) -> None:
        development = self._sample_shots(80, game_id=1)
        prediction = self._sample_shots(20, game_id=2, include_target=False)
        prediction.index = range(100, 120)

        models = fit_final_models(
            development,
            context_iterations=30,
            identity_iterations=20,
        )
        probabilities = predict_final_probabilities(
            models,
            development,
            prediction,
            minimum_identity_history=5,
        )

        self.assertEqual(probabilities.index.tolist(), prediction.index.tolist())
        self.assertEqual(len(probabilities), len(prediction))
        self.assertTrue(probabilities["history_aware_probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
