# Caputo LSTM — COVID-19 Case Prediction

A from-scratch implementation of an LSTM cell using the Caputo fractional derivative in the backpropagation step, tested on real WHO COVID-19 data.

## What is this?

Standard LSTMs use integer-order derivatives during backpropagation. This project replaces them with the **Caputo fractional derivative** (order σ), which introduces a memory effect over the gradient history. The goal is to test whether fractional-order backpropagation affects learning on epidemiological time series.

## Data

WHO COVID-19 global dataset. Download from:
https://data.who.int/dashboards/covid19/data

Download The .csv file named as:
Daily frequency reporting of new COVID-19 cases and deaths by date reported to WHO

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