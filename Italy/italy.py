from CaputoLstm import LSTMCell
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

########################################### SET THE DATA SET ###########################################
df = pd.read_csv("WHO_data.csv")

italy = df[df["Country"] == "Italy"].copy() # copy just for safety to protect original data,nothing else
italy = italy[["Date_reported", "New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]] # only keep usefull columns
italy["Date_reported"] = pd.to_datetime(italy["Date_reported"]) # convert text date data into actual date format pandas understands

# print(italy.head(20)) #first row date is "2020-01-31"
# print(italy.tail(20)) 

italy = italy[italy["New_cases"] >= 0] #!

########## Data gap sequence dates ##########
#print(italy["Date_reported"].diff().value_counts())
# Date_reported
# 22 days        1 "2020-01-31" / "2020-02-22"
# 1 days      1346
# after "2023-11-06"
#[7 days        94 
# 175 days       1]


italy = italy[italy["Date_reported"] > "2020-01-31"] # "2020-01-31" is the first day where FIRST CASE come out
italy = italy[italy["Date_reported"] < "2023-11-06"] # "2023-11-06" is the first day of the non-daily data rows


italy = italy.sort_values("Date_reported").reset_index(drop=True) # reset index of the data table

########################################### NORMALIZATION ###########################################
#print(italy.describe()) #Max is 11 times of the mean so yes we need normalization

italy["New_cases"] = np.log1p(italy["New_cases"]) # log(x+1) for x = 0 protection

# Before:
# mean:  19,500
# max:  228,123
# std:   31,493

# After
# mean:  8.77
# max:  12.34
# std:   1.68

# 80 / 20 rule (1347 rows)
split = int(len(italy) * 0.8)
train_data = italy.iloc[:split] # %80 (1077 rows)
test_data = italy.iloc[split:]  # %20 (270 rows)
print(len(train_data), len(test_data))

# windowing 
window_x_seq = []
window_y_seq = []

for i in range(len(train_data)-6):
    window = train_data["New_cases"].iloc[i:i+7]
    x = window.iloc[:6]
    y = window.iloc[6]
    window_x_seq.append(x)
    window_y_seq.append(y)


# our cell expects np arrays
window_x_seq = np.array(window_x_seq)
window_y_seq = np.array(window_y_seq)
