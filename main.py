from CaputoLstm import LSTMCell
import pandas as pd


df = pd.read_csv("data.csv")

turkey = df[df["Country"] == "Türkiye"].copy() # copy just for safety to protect original data,nothing else
turkey = turkey[["Date_reported", "New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]] # only keep usefull columns
turkey["Date_reported"] = pd.to_datetime(turkey["Date_reported"]) # convert text date data into actual date format pandas understands

turkey = turkey[turkey["New_cases"] >= 0] #!

turkey = turkey.sort_values("Date_reported").reset_index(drop=True) # reset index of the data table


print(turkey.shape)
print(turkey.head())
print(turkey.describe()) # we decided normalization with data values gap diffrence
