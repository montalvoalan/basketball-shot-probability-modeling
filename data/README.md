# Data

This project uses season-level `shotdetail` Parquet files from the public [cdechoch/nba-data-archive](https://huggingface.co/datasets/cdechoch/nba-data-archive) mirror on Hugging Face.

## Source and license

- Dataset: `cdechoch/nba-data-archive`
- Dataset revision: `73fb165135d4d1ef8daaad26fb5b9a29b9d2bdee`
- Dataset-card license: Apache-2.0
- Upstream archive: [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)
- Original shot-detail source identified by the archive: NBA.com

The project attributes both the mirror and upstream archive. It does not redistribute the data files; the download script retrieves the selected files directly from the pinned public revision.

## Selected files

The year in each filename represents the starting year of the NBA season.

| File | NBA season | Rows | Make rate | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `2021.parquet` | 2021-22 | 216,722 | 46.11% | `875bf67a77f0e9d69c26ceae966c28d82de7c31f8f265c518ddb8fbb897fa6c8` |
| `2022.parquet` | 2022-23 | 217,218 | 47.54% | `ebea41a1c5364f688c33c354b9e9780dfd8ae34c6fdfeadd5d9bc28cad9beba3` |
| `2023.parquet` | 2023-24 | 218,701 | 47.43% | `1c058cbdb645d4e3ae16ac80d4635d1279d216b6c6490a52bd89929f6d364c54` |
| `2024.parquet` | 2024-25 | 219,527 | 46.72% | `20cdc43709ba8f4cbe80071c62f377f7c8f5b0b135168cc94545dfba933fba5e` |

The initial source audit found no missing `SHOT_MADE_FLAG` values and no duplicate `(GAME_ID, GAME_EVENT_ID)` keys in these files.

## Modeling split

- Development seasons: 2021-22 through 2023-24
- Temporal validation: rolling or expanding season folds within the development period
- Final untouched test: 2024-25

The final test season will not be used for feature selection, model selection, early stopping, calibration decisions, or hyperparameter choices.

## Minimum useful fields

The selected files contain:

- game and event identifiers;
- binary shot outcome;
- season and game date;
- action type and two- or three-point classification;
- shot zones, distance, and court coordinates;
- period and game-clock time remaining;
- player and team identifiers and names;
- home and visiting team abbreviations.

The data does not contain defender proximity, shooter speed, dribble count, or the possession shot clock. Those concepts will not be fabricated or implied in the analysis.

`EVENT_TYPE` directly states made versus missed and must be excluded because it duplicates the target. `SHOT_MADE_FLAG` is the target. Game/event IDs and constant fields will be used for integrity checks but not as predictors.

## Download

From the repository root:

```bash
python src/download_data.py
```

The script downloads the four pinned files to `data/raw/`, validates file size and SHA-256, and skips files that already match.

## Version-control policy

Raw and processed data files are ignored by Git. Only source documentation and reproducible download code are committed.

No files from the private internship assessment belong in this repository.
