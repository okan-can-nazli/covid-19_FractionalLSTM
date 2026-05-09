# Caputo LSTM — COVID-19 Case Prediction

A from-scratch implementation of an LSTM cell using the Caputo fractional derivative in the backpropagation step, tested on real WHO COVID-19 data.

## What is this?

Standard LSTMs use integer-order derivatives during backpropagation. This project replaces them with the **Caputo fractional derivative** (order σ), which introduces a memory effect over the gradient history. The goal is to test whether fractional-order backpropagation affects learning on epidemiological time series.

## Project Structure

```
├── CaputoLstm.py              # Caputo LSTM cell (from scratch, no ML libraries)
├── main.py                    # Single sigma training + testing
├── analysis_main.py           # Multi-sigma comparison loop [1.0, 0.9, 0.8, 0.7]
├── Türkiye_data/
│   ├── turkey_set_data.py     # Data pipeline for Turkey
│   └── Türkiye Results/
│       ├── standart analysis/ # Stateless LSTM, sigma [1.0, 0.9, 0.8, 0.7]
│       └── Stateful analysis/ # Stateful LSTM experiment
└── Italy_data/
    ├── italy_set_data.py      # Data pipeline for Italy
    └── Italy Results/
        ├── 2023 excluded/     # Results without 2023 data
        └── 2023 included/     # Results with 2023 data
```

## Data

WHO COVID-19 global dataset. Download from:
https://data.who.int/dashboards/covid19/data

Download the .csv file named:
**Daily frequency reporting of new COVID-19 cases and deaths by date reported to WHO**

Save as `WHO-COVID-19-global-daily-data.csv` in the project root.

## How to Run

**Single sigma:**
```bash
python main.py
```

**Multi-sigma comparison:**
```bash
python analysis_main.py
```

To switch country, change the import at the top of either file:
```python
from Türkiye_data.turkey_set_data import get_data
# or
from Italy_data.italy_set_data import get_data
```

## Key Findings

- Stable sigma range: **0.7 – 1.0**. Below 0.7 the fractional memory accumulates too aggressively and training diverges.
- Sigma=1.0 (standard LSTM) performs best on test data for Turkey.
- Fractional sigmas (0.8, 0.9) are competitive on training and still converging at 5000 epochs while sigma=1.0 had plateaued.
- The cell generalizes across two countries with different epidemiological patterns (Turkey and Italy).
- Test loss is higher than train loss in both countries due to **out-of-distribution behavior** — the Omicron wave (Turkey) and the post-pandemic near-zero period (Italy) — not a cell failure.
- Stateful LSTM performed significantly worse than stateless — test loss increased from 0.22 to 36.0.

## Constants

| Parameter | Value |
|-----------|-------|
| STM size | 16 |
| Window size | 6 days → predict day 7 |
| Epochs | 5000 |
| Learning rate | 0.01 |
| Random seed | 42 |
| Normalization | log1p / expm1 |

---

## Results

### 🇹🇷 Turkey (σ = 1.0, 5000 epochs)

**Training**
![Turkey Train](Türkiye_data/Türkiye%20Results/standart%20analysis/Train%20results/Train_sigma_1.0.png)

**Test**
![Turkey Test](Türkiye_data/Türkiye%20Results/standart%20analysis/Test%20resuts/Test_sigma_1.0.png)

**Terminal Output Summary**
```
Final Train Loss: 0.01667
Final Test Loss:  0.22626

sigma=1.0 → Train: 0.01667 | Test: 0.22626
sigma=0.9 → Train: 0.01835 | Test: 0.23572
sigma=0.8 → Train: 0.01839 | Test: 0.25655
sigma=0.7 → Train: 0.01698 | Test: 0.26146
```

---

### 🇮🇹 Italy — 2023 Excluded (σ = 0.9, 5000 epochs)

**Training**
![Italy Train](Italy_data/Italy%20Results/2023%20excluded/Train%20results/Train_sigma_0.9.png)

**Test**
![Italy Test](Italy_data/Italy%20Results/2023%20excluded/Test%20results/Test_sigma_0.9.png)

**Terminal Output Summary**
```
Final Train Loss: 0.08934
Final Test Loss:  0.34290

sigma=1.0 → Train: 0.08445 | Test: 0.50587
sigma=0.9 → Train: 0.08934 | Test: 0.34290
sigma=0.8 → Train: 0.08674 | Test: 0.39678
sigma=0.7 → Train: 0.08728 | Test: 0.40454
```