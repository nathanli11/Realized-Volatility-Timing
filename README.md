# Realized Volatility Timing

This repository contains a final project on **realized volatility timing** for option carry strategies.
The core idea is to estimate realized volatility with a **Heston state-space model** and an
**Unscented Kalman Filter (UKF)**, then use the implied-realized volatility spread to dynamically
scale an option strategy.

## Project Overview

The final notebook combines two blocks:

1. **Option strategy backtesting**
   - Short strangle baseline strategy
   - Comparison with and without transaction costs
   - Delta hedging
   - Delta-gamma hedging

2. **Realized volatility timing**
   - Rolling MLE calibration of Heston parameters
   - UKF filtering of latent variance
   - Construction of the spread
     - `s_t = sigma_IV,t - sigma_hat_t`
   - Dynamic timing/allocation based on the spread
   - Out-of-sample checks and robustness analysis

## Main Notebook

The main deliverable is:

- [notebooks/realized_vol_timing.ipynb](notebooks/realized_vol_timing.ipynb)

It includes:

- data loading and preprocessing
- backtests of option strategies
- rolling Heston-UKF calibration
- no-look-ahead timing logic
- robustness checks
- a section to run unit tests from the notebook

## Repository Structure

```text
investment_lab/
  backtest.py
  heston_ukf.py
  option_trade.py
  option_strategies.py
  rates.py
  data/
notebooks/
  realized_vol_timing.ipynb
tests/
  test_heston_ukf.py
requirements.txt
README.md
```

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Running the Project

Open the main notebook:

```bash
jupyter notebook notebooks/realized_vol_timing.ipynb
```

Recommended execution order:

1. run the notebook from top to bottom
2. let the rolling calibration cache build in `.cache/heston_ukf/`
3. execute the robustness section
4. execute the unit-test section at the end

## Unit Tests

Targeted unit tests are available for the UKF timing pipeline:

```bash
pytest -q tests/test_heston_ukf.py
```

These tests cover:

- signal construction and clipping
- execution lag to avoid look-ahead bias
- alignment of rolling-filter outputs
- spread construction
- propagation of timing parameters through the end-to-end pipeline

## Methodological Notes

- Heston parameters are calibrated on a **rolling window**
- the timing signal is executed with a **1 business-day lag**
- this avoids using future information in the backtest
- robustness is checked across multiple calibration windows and signal rules

## Disclaimer

- This repository is for **educational purposes only**
- It does **not** constitute investment advice
- Results depend on modeling assumptions, data quality, and execution conventions

## License

This project is licensed under the terms specified in this repository.
