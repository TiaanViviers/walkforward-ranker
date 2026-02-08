# walkforward-ranker

A walk-forward evaluation framework for ranking-based decision systems under non-stationarity, with emphasis on reproducibility and diagnostics.

## Overview

**walkforward-ranker** is a general walk-forward evaluation framework for time-indexed decision problems where, at each decision time, a system must rank a discrete set of candidate actions using machine learning models and select one or more actions to execute.

This framework is designed for problems where:
- Decisions are made at regular intervals (e.g., daily)
- A discrete set of candidate actions is available at each decision point
- Realized outcomes are observed after-the-fact
- The environment is non-stationary (patterns change over time)

**What this is:** An ML experiment harness for developing, evaluating, and stress-testing ranking-based systems under realistic walk-forward constraints.

**What this is not:** A trading strategy, data pipeline, or backtester. Domain-specific logic lives outside this framework.

## Key Features

- **Walk-forward validation**: Rolling and expanding window support with configurable retraining schedules
- **Ranking-based ML**: Built-in support for LightGBM LambdaRank with extensible model interface
- **Feature selection**: Correlation-based and importance-based feature selection pipeline
- **Reproducibility**: Every run is fully logged with config, features, and model artifacts
- **Production-ready**: Clean separation between research and inference workflows
- **Diagnostics-first**: Comprehensive evaluation metrics and time-localized performance analysis

## Project Structure

```
walkforward-ranker/
├── config/                    # Configuration files
│   ├── example_config.yaml    # Example configuration with dummy features
│   └── schema.yaml           # Configuration schema documentation
│
├── data/                     # Data directory (gitignored)
│
├── models/                   # Saved models (gitignored)
│
├── results/                  # Results and metrics (gitignored)
│
├── src/                      # Core framework code
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # Data loading utilities
│   ├── evaluator.py         # Evaluation metrics
│   ├── feature_registry.py  # Feature ordering management
│   ├── feature_selector.py  # Feature selection pipeline
│   ├── model_artifacts.py   # Model saving/loading
│   ├── ranker.py           # LightGBM ranking model
│   ├── selector.py         # Action selection policies
│   └── splitter.py         # Walk-forward splitting
│
├── scripts/                 # Execution scripts
│   ├── 01_feature_selection.py  # One-time feature selection
│   ├── 02_train.py             # Walk-forward training
│   ├── 03_predict.py           # Production inference
│   └── 04_evaluate.py          # Results analysis
│
└── notebooks/              # Jupyter notebooks for exploration
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/walkforward-ranker.git
cd walkforward-ranker

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:

```
data/
├── calibration/
│   └── {asset}_calibration.parquet       # 10% - for feature selection
├── replay_window/
│   └── {asset}_replay_window.parquet     # 75% - for walk-forward training
└── holdout/
    └── {asset}_holdout.parquet           # 15% - for final testing
```

Each file should be in long-form format with one row per (timestamp, action) pair:

```
| date       | group_id | action_param_1 | action_param_2 | feature_1 | feature_2 | ... | pnl    |
|------------|----------|----------------|----------------|-----------|-----------|-----|--------|
| 2023-01-03 | 0        | 0.005          | 0.010          | 12.3      | 45.6      | ... | 150.0  |
| 2023-01-03 | 0        | 0.007          | 0.012          | 12.3      | 45.6      | ... | -80.0  |
```

Required columns:
- `date`: Timestamp for decision point
- `group_id`: Group identifier (groups rows that form a ranking problem)
- Action features (parameters that vary per action)
- Context features (features constant within a group)
- `pnl`: Realized outcome (label)

### 2. Create Configuration

Copy the example config and update with your settings:

```bash
cp config/example_config.yaml config/production_config.yaml
# Edit production_config.yaml with your feature names and parameters
```

### 3. Feature Selection

Run one-time feature selection on calibration data:

```bash
python scripts/01_feature_selection.py --asset us30 --config config/production_config.yaml
```

This will:
- Load calibration data (10% sample)
- Remove low-variance features
- Remove highly correlated features
- Select features by importance
- Save asset-specific feature registry

### 4. Walk-Forward Training

Train model on replay window data:

```bash
python scripts/02_train.py --asset us30 --config config/production_config.yaml
```

This will:
- Load replay window data (75% of data)
- Split into walk-forward windows
- Train and evaluate on each window
- Save final model with metadata

### 5. Run Complete Pipeline

Or run everything at once:

```bash
# Full pipeline
python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml

# Specific stages
python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml --stages feature_selection
python scripts/run_pipeline.py --asset us30 --config config/production_config.yaml --stages training,evaluation
```

### 6. Production Inference

Use trained model for prediction:

```bash
python scripts/03_predict.py --model-id us30_20260205_103000 --input data/today.parquet --output predictions/today.csv
```

### 7. Analyze Results

Evaluate walk-forward results:

```bash
python scripts/04_evaluate.py --run-id us30_20260205_103000
```

### 8. Clean Generated Artifacts

Remove generated run artifacts when `models/`, `results/`, and cache folders get large:

```bash
# Preview what would be removed
python scripts/clean_generated.py --all --dry-run

# Remove generated artifacts
python scripts/clean_generated.py --all
```

Optional shortcuts:

```bash
make clean-dry
make clean
```

## Configuration

Key configuration options:

```yaml
data:
  path: "data/preprocessed_data.parquet"
  date_col: "date"
  group_col: "group_id"
  label_col: "pnl"
  config_features: [...]    # Action parameters
  market_features: [...]    # Context features
  calendar_features: [...]  # Calendar dummies

walkforward:
  initial_train_days: 365
  train_window_days: 365
  test_window_days: 30
  retrain_frequency_days: 7
  expanding_window: false   # true = expanding, false = rolling

model:
  objective: "lambdarank"
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 100
  random_state: 42

selection:
  top_k: 3  # Number of actions to select

feature_selection:
  correlation_threshold: 0.95
  min_importance_percentile: 5
```

See [config/schema.yaml](config/schema.yaml) for full documentation.

## Evaluation Metrics

The framework computes multiple ranking-quality metrics:

- **NDCG@K**: Normalized discounted cumulative gain
- **Hit Rate@K**: Proportion of times top-K contains a good action
- **Precision@K**: Proportion of predicted top-K that are actually top-K
- **Regret**: Missed reward vs. optimal selection

All metrics are computed per time period for temporal analysis.

## Model Artifacts

Each training run saves:
```
models/{run_id}/
  ├── model.pkl              # Trained model
  ├── feature_list.json      # Ordered feature names
  ├── config.yaml           # Configuration used
  └── metadata.json         # Metrics and metadata
```

This ensures full reproducibility: every prediction can be traced back to the exact model, features, and configuration.

## Privacy and Security

This repository is designed to keep sensitive data private:

- All data files are gitignored
- Production configs with real feature names are gitignored
- Example configs use dummy feature names
- Code references generic column names (date, pnl, group_id)

See [.gitignore](.gitignore) for details.

## Design Principles

1. **Strategy-agnostic**: No embedded domain logic
2. **Non-stationarity aware**: Walk-forward is the default
3. **Separation of concerns**: Data generation ≠ ML ≠ evaluation
4. **Diagnostics over blind optimization**: Understand when and why models work
5. **Production-ready**: Single script for inference, full artifact tracking

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This is a personal research project. Issues and suggestions are welcome, but please note this is tailored to specific use cases.

## Citation

If you use this framework in your research, please cite:

```
@software{walkforward_ranker,
  title={walkforward-ranker: A Walk-Forward Evaluation Framework for Ranking Systems},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/walkforward-ranker}
}
```
