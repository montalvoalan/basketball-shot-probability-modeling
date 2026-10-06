# Basketball Shot Probability Modeling

## Overview

This project builds a probability model for NBA shot outcomes using shot context and court-location data. It compares an interpretable logistic regression baseline with a CatBoost model that can capture nonlinear relationships and categorical features.

## Data

The project uses regular-season shot-detail data from the public [NBA Data Archive](https://huggingface.co/datasets/cdechoch/nba-data-archive).

| Season | Role | Shots |
| --- | --- | ---: |
| 2021-22 | Training and development | 216,722 |
| 2022-23 | Training and development | 217,218 |
| 2023-24 | Temporal validation | 218,701 |
| 2024-25 | Final holdout | 219,527 |

Raw data is not stored in this repository. Source details and expected files are documented in [`data/README.md`](data/README.md).

## Modeling approach

- Validate the row grain, target, identifiers, missing values, and feature ranges.
- Use a season-based split instead of a random split.
- Establish constant-probability and regularized logistic regression baselines.
- Train CatBoost on numerical and categorical shot-context features.
- Compare models using log loss, Brier score, and calibration.
- Evaluate performance across shot distance, action type, court zone, and game-clock context.

The final 2024-25 season will remain untouched until model and feature decisions are complete.

## Tools

Python, pandas, scikit-learn, CatBoost, Matplotlib, Seaborn, and Jupyter.

## Current status

The data source and validation design are complete. Data understanding and exploratory analysis are next.
