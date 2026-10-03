# Time-series forecasting

**Status:** planned

## Research question

Which forecasting approaches (naive and seasonal baselines, gradient boosting with lags, statistical models, deep models) win on business demand series, under honest forward-in-time validation?

## Why it matters for DCLab

Demand, capacity and revenue forecasts are core DCLab use cases, and time leakage is the easiest way to fool a forecast.

## Leakage traps specific to this field

- **Look-ahead features:** lags or rolling windows that include the current or future period.
- **Random splits:** in the expansion campaign, random K-fold reported MAE 430 vs 833 for time-ordered CV on bike sharing (EXP-062).
- **Target components:** columns that sum to the target (casual + registered = count on bike sharing).

## First experiments

| ID | Question |
|---|---|
| TS-001 | Seed result, done: bike sharing daily. XGBoost MAE 680 vs 1,156 for the lag-2 naive baseline (`evidence/campaigns/expansion_v1`, EXP-061..065) |
| TS-002 | Hourly bike demand and Rossmann store sales: gradient boosting with lags vs ETS/ARIMA vs a global deep model, on rolling-origin backtests |
| TS-003 | How large must a backtest be (number of origins) before model rankings stabilize? |

## Candidate datasets

Bike sharing (UCI), Rossmann store sales and M5 (Kaggle), electricity load (UCI).

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=time-series-forecasting TITLE="Time-series forecasting" PREFIX=TS
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
