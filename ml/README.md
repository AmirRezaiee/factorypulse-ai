# FactoryPulse AI — Manufacturing Quality Forecasting

**Independent portfolio project · Synthetic solar-module manufacturing data · not actual factory production data**

## Objective
Predict each line's next-day repair rate (%) using observations available at the end of the preceding production day. Lines E4–E7 are simulated.

## Methods
- Daily aggregation of module-level synthetic records.
- Historical features: current-day repair rate, prior-day rate, 3-/7-day rolling rates, 7-day rolling production volume, line ID, and next-day weekday.
- Chronological 80/20 holdout, shared cutoff across lines. The target is checked to correspond to the *next calendar day*.
- Random Forest regressor, compared with (1) yesterday's repair rate and (2) each line's training-period historical mean.
- Four additional expanding-window evaluations. Training always precedes evaluation dates.

## Reproduce
```bash
python -m pip install -r requirements.txt
python train.py
```

## Results (held-out dates September 12–28, 2026; 68 line-days)
See `outputs/metrics.json` for full precision and per-line scores.

| Method | MAE (percentage points) | RMSE (pp) | R² |
|---|---:|---:|---:|
| Random Forest | 4.29 | 5.26 | -0.324 |
| Yesterday's repair rate | 5.15 | 6.75 | -1.185 |
| Historical line mean | See metrics.json | See metrics.json | See metrics.json |

**Interpretation:** RF beats the persistence baseline on this holdout, but negative R² indicates weak predictive fit. In four expanding-window tests, RF beat the historical-line-mean baseline on only two folds. The project does **not** demonstrate reliable operational forecasting or deployed predictive maintenance.

## Files
- `train.py`: reproducible feature construction, modeling, holdout and backtesting.
- `data/fact_module_quality.csv`: synthetic module-level source data.
- `outputs/metrics.json`: holdout results, including both baselines.
- `outputs/backtests.json`: four expanding-window evaluations.
- `outputs/holdout_predictions.csv`: model predictions and baselines.
- `outputs/actual_vs_predicted.png`: visualization.

## Limitations and future work
- Synthetic data may lack the time dependencies, class imbalance, and distribution shifts of real factories.
- The current feature set may be insufficient; equipment alarms, recipes, material lots, and maintenance events would require properly authorized data.
- The rolling evaluation folds have distinct evaluation periods, but their expanding training windows overlap.
- Hyperparameter tuning, drift tests, and calibrated uncertainty estimates are not yet implemented.
- Do not infer causality or real-world equipment-health performance from these simulations.
