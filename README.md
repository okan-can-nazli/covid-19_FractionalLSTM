## Data

WHO COVID-19 global dataset. Download from:
https://covid19.who.int/data

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