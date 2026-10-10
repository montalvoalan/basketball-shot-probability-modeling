# Methodology and Decision Rationale

## Objective

The goal is to estimate the probability that an NBA field-goal attempt is made using information available before the result. The emphasis is probability quality and season-to-season reliability, not maximizing a classification accuracy threshold.

This framing supports questions such as whether a group of attempts was expected to be efficient, how much shot context explains outcomes, and whether player identity adds reliable predictive information.

## Data contract

The project uses 872,168 regular-season shots from the public NBA Data Archive:

- 2021-22 through 2023-24: 652,641 development shots.
- 2024-25: 219,527-shot final holdout.
- Target: `SHOT_MADE_FLAG`.
- Row grain: one field-goal attempt.
- Audit key: the combination of `GAME_ID` and `GAME_EVENT_ID`.

`EVENT_TYPE` is excluded because it directly states whether a shot was made or missed. Identifiers and constant source fields are retained only for auditing unless explicitly tested as identity features.

## Why temporal validation

A random train-test split would allow shots from the same season, teams, and player situations to appear on both sides of the evaluation. That setting is easier than the intended use case: estimating probabilities for a future season.

The project therefore used expanding temporal folds:

1. Train on 2021-22 and validate on 2022-23.
2. Train on 2021-22 and 2022-23 and validate on 2023-24.
3. Lock every decision, refit on all three development seasons, and evaluate once on 2024-25.

The final season did not influence feature selection, model selection, iteration counts, the player-history rule, or calibration.

## Features

The context model uses 16 features derived consistently for every season.

Numeric features include reported shot distance, squared distance, court coordinates in feet, absolute lateral distance, shot angle, period, seconds remaining in the period, a final-five-seconds indicator, overtime, and corner-three location.

Categorical features include action type, two- or three-point shot type, basic zone, zone area, and zone range. The identity experiment adds player and team IDs as categorical variables.

All features are available before the shot outcome. The project does not create substitutes for unavailable tracking variables such as defender distance or dribble count.

## Model choices

### Constant benchmark

The training make rate establishes how well a model performs when it assigns the same probability to every shot. It is useful for verifying that more complex models add shot-level information.

### Logistic regression

L2-regularized logistic regression provides an interpretable statistical baseline. Numeric scaling and categorical one-hot encoding are fitted inside each temporal fold, and unseen categories are ignored safely.

### CatBoost

CatBoost was selected because the data combines nonlinear spatial relationships with categorical basketball context. It can model interactions such as different distance effects across shot types without manually creating every interaction or expanding high-cardinality categories into a very large sparse matrix.

Only a small, predeclared depth comparison was used. Depth 5 was retained because depth 7 did not provide enough validation improvement to justify added complexity. Final iteration counts were fixed from development-only evidence: 1,200 for the context model and 612 for the identity model.

## The history-aware rule

Player and team identity slightly improved overall development log loss, but raw identity harmed players who were unseen or had limited prior attempts. The final design therefore uses:

- Context probability when a player has fewer than 250 development attempts.
- Identity probability when a player has at least 250 development attempts.

The 250-attempt boundary came from a pre-existing diagnostic group, not a broad threshold search. It improved log loss and calibration in both development folds. Temporal Platt calibration was tested and rejected because it slightly worsened log loss, Brier score, and ECE on the latest development fold.

## Evaluation metrics

- **Log loss** is the primary metric. It rewards accurate probabilities and strongly penalizes confident mistakes.
- **Brier score** is the mean squared error of the predicted probabilities.
- **Expected calibration error (ECE)** summarizes the gap between average predictions and observed make rates across ten probability bins.

No one metric tells the full story. A constant league-average prediction can have low aggregate ECE while failing to distinguish easy and difficult shots, which is why log loss and Brier score are evaluated alongside calibration.

## Final results

| Model | Log loss | Brier score | ECE |
| --- | ---: | ---: | ---: |
| CatBoost identity | **0.63401** | **0.22283** | 0.01027 |
| CatBoost history-aware | 0.63419 | 0.22291 | 0.00457 |
| CatBoost context | 0.63586 | 0.22366 | **0.00368** |
| Logistic regression | 0.64037 | 0.22556 | 0.00724 |
| Constant benchmark | 0.69101 | 0.24893 | 0.00307 |

The selected history-aware model improved holdout log loss by 0.00618, or 0.97%, over logistic regression. It also improved by 0.00167 over context-only CatBoost.

Raw identity CatBoost finished 0.00018 ahead of the history-aware model on holdout log loss. However, its ECE was 0.01027 compared with 0.00457 for the selected model. The history-aware model remains the final selection because it was chosen before the holdout was opened. Replacing it afterward would turn the final test into another model-selection set.

## Diagnostics and interpretation

Common distance, zone, and clock segments were generally well calibrated during development. The most visible remaining action-level weaknesses were overprediction for hook shots and putback layups. Category-specific corrections were rejected because they would require another future season to validate safely.

On the final season, raw identity unexpectedly improved log loss for unseen and low-history players. That reversal is a useful result: the predictive value of identity can change with roster turnover, role changes, player development, injuries, and changes in shot selection.

## Limitations and future work

The model lacks tracking and richer game-state variables, so its probabilities should not be interpreted as a complete measure of shot quality or shooter skill. It is observational rather than causal, and it does not separate player ability from the offensive situations in which attempts occur.

Future work should be evaluated on a new out-of-time season. Useful extensions include defender and movement data, hierarchical player effects that shrink limited-history players toward a population estimate, and uncertainty intervals around player- or team-level summaries.

## Reproducibility and AI assistance

Reusable feature, modeling, diagnostic, and final-workflow functions live in `src/`; their core contracts are covered by automated tests. Executed notebooks preserve the analysis sequence and outputs. Raw data, trained models, and row-level predictions are excluded from Git but can be regenerated from the documented inputs.

AI tools were used as a development assistant for structured planning, code scaffolding and review, test design, workflow automation, and documentation refinement. The modeling evidence comes from the executed temporal-validation workflow, and no model changes were made after the final holdout was evaluated.
