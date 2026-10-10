"""Build the public Phase 8 figures from saved final-workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import calibration_table  # noqa: E402


METRICS_PATH = PROJECT_ROOT / "outputs" / "final_holdout_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "final_holdout_predictions.parquet"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "raw" / "2024.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_COLORS = {
    "Constant benchmark": "#9CA3AF",
    "Logistic regression": "#4C78A8",
    "CatBoost context": "#F58518",
    "CatBoost identity": "#54A24B",
    "CatBoost history-aware": "#E45756",
}


def _load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the saved final metrics, predictions, and holdout rows."""
    required_paths = [METRICS_PATH, PREDICTIONS_PATH, HOLDOUT_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Run notebooks/06_final_workflow.ipynb before building figures. "
            f"Missing: {missing_paths}"
        )

    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics_payload = json.load(metrics_file)
    metrics = pd.DataFrame(metrics_payload["metrics"])
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    holdout = pd.read_parquet(HOLDOUT_PATH)

    expected_keys = ["GAME_ID", "GAME_EVENT_ID"]
    if len(predictions) != len(holdout):
        raise ValueError("Prediction and holdout row counts do not match")
    if not predictions[expected_keys].reset_index(drop=True).equals(
        holdout[expected_keys].reset_index(drop=True)
    ):
        raise ValueError("Prediction and holdout event order does not match")
    if not set(holdout["SHOT_MADE_FLAG"].unique()).issubset({0, 1}):
        raise ValueError("Holdout target must be binary")

    return metrics, predictions, holdout


def _save_model_comparison(metrics: pd.DataFrame) -> None:
    """Plot final holdout log loss for each benchmark and model."""
    plot_data = metrics.sort_values("log_loss", ascending=False).copy()
    colors = [MODEL_COLORS[model] for model in plot_data["model"]]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(plot_data["model"], plot_data["log_loss"], color=colors)
    ax.set_xlim(0.62, 0.70)
    ax.set_xlabel("Log loss (lower is better)")
    ax.set_ylabel("")
    ax.set_title("Final 2024-25 holdout performance", loc="left", weight="bold")
    ax.bar_label(
        bars,
        labels=[f"{value:.5f}" for value in plot_data["log_loss"]],
        padding=4,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_calibration(predictions: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Plot reliability curves for the three CatBoost variants."""
    target = holdout["SHOT_MADE_FLAG"].to_numpy()
    probability_columns = {
        "CatBoost context": "context_probability",
        "CatBoost identity": "identity_probability",
        "CatBoost history-aware": "history_aware_probability",
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#6B7280",
        label="Perfect calibration",
    )
    for model_name, column in probability_columns.items():
        table = calibration_table(target, predictions[column].to_numpy()).query("shots > 0")
        ax.plot(
            table["average_prediction"],
            table["observed_make_rate"],
            marker="o",
            linewidth=2,
            label=model_name,
            color=MODEL_COLORS[model_name],
        )

    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.15, 0.85)
    ax.set_xlabel("Average predicted make probability")
    ax.set_ylabel("Observed make rate")
    ax.set_title("Final 2024-25 calibration", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _binary_log_loss(target: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))


def _save_player_history(predictions: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Compare CatBoost variants across development-time player volume groups."""
    history_group = pd.cut(
        predictions["training_player_attempts"],
        bins=[-1, 0, 249, 749, np.inf],
        labels=["Unseen", "1-249", "250-749", "750+"],
    )
    target = holdout["SHOT_MADE_FLAG"].to_numpy()
    model_columns = {
        "Context": "context_probability",
        "Identity": "identity_probability",
        "History-aware": "history_aware_probability",
    }
    records = []
    for group in history_group.cat.categories:
        group_mask = (history_group == group).to_numpy()
        for model_name, probability_column in model_columns.items():
            records.append(
                {
                    "Player history": str(group),
                    "Model": model_name,
                    "Log loss": _binary_log_loss(
                        target[group_mask],
                        predictions.loc[group_mask, probability_column].to_numpy(),
                    ),
                }
            )
    plot_data = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.barplot(
        data=plot_data,
        x="Player history",
        y="Log loss",
        hue="Model",
        palette={
            "Context": "#F58518",
            "Identity": "#54A24B",
            "History-aware": "#E45756",
        },
        ax=ax,
    )
    ax.set_ylim(0.61, 0.65)
    ax.set_xlabel("Player attempts in the development seasons")
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Final performance by prior player history", loc="left", weight="bold")
    ax.legend(title="", frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "player_history_log_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create every public report figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics, predictions, holdout = _load_artifacts()
    _save_model_comparison(metrics)
    _save_calibration(predictions, holdout)
    _save_player_history(predictions, holdout)
    print(f"Saved 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
