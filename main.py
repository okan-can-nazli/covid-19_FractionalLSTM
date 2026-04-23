from CaputoLstm import LSTMCell
import pandas as pd
import numpy as np

########################################### SET THE DATA SET ###########################################
df = pd.read_csv("WHO_data.csv")

turkey = df[df["Country"] == "Türkiye"].copy() # copy just for safety to protect original data,nothing else
turkey = turkey[["Date_reported", "New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]] # only keep usefull columns
turkey["Date_reported"] = pd.to_datetime(turkey["Date_reported"]) # convert text date data into actual date format pandas understands

turkey = turkey[turkey["New_cases"] >= 0] #!


# #print(turkey.describe()) # we decided normalization with data values gap diffrence


# Turkey switched from case-based to aggregate reporting after 2022
# Post-2023 data is near-zero with no death records — not usable for training

# print(turkey[turkey["Date_reported"] > "2023-01-01"].head(20)) 
# print(turkey[turkey["Date_reported"] < "2023-01-01"].tail(40)) 


turkey = turkey[turkey["Date_reported"] < "2023-01-01"]



# print(turkey["Date_reported"].diff().value_counts())
# gaps = turkey[turkey["Date_reported"].diff() == "7 days"]["Date_reported"]
# print(gaps)

# gaps = turkey[turkey["Date_reported"].diff() == "2 days"]["Date_reported"]
# print(gaps)


# March 2020 → June 2022    →  daily reporting   (807 rows)
# 2022-06-05 first day of weekly data rows start
# June 2022  → November 2022 →  weekly reporting  (24 rows)
# First two rows (2020-03-12, 2020-03-14) have a 2-day gap — artifact of pandemic start, not a reporting issue


turkey = turkey[turkey["Date_reported"] < "2022-06-05"] # remove data on weekly rows
turkey = turkey.iloc[1:] # remove first day data (2 day gap)

turkey = turkey.sort_values("Date_reported").reset_index(drop=True) # reset index of the data table

########################################### NORMALİZATİON ###########################################
turkey["New_cases"] = np.log1p(turkey["New_cases"]) # log(x+1) for x = 0 protection
# Before:   min=0,  max=406,321,  std=28,954
# After:    min=0.69,  max=11.6,  std=1.36



########################################### TRAİN ###########################################
# 80 / 20 rule (807 rows)
train_data = turkey.iloc[:646] # %80 (646 rows)
test_data = turkey.iloc[646:]  # %20 (161 rows)
# print(len(train_data), len(test_data))

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



#! CONSTANTS
np.random.seed(42)
SİGMA = 1
STM_SİZE = 16
epochs = 1000

# init cell
cell = LSTMCell(input_size = 1,stm_size = STM_SİZE,output_size = 1) #! input_size means multivariate data!!!

print(SİGMA)

for step in range(epochs):
    
    total_loss = 0
    
    window_outputs = []
    dy_preds = []
    window_caches = []
    
    # forward
    for i, window in enumerate(window_x_seq): 
        window_stm_outputs, window_caches = cell.forward_sequence(x_sequence = window, stm_init = np.zeros((STM_SİZE,1)), ltm_init = np.zeros((STM_SİZE,1)))
                
        
        prediction = cell.predict(window_stm_outputs[-1])  # (16,1) → (1,1)
        error = prediction - window_y_seq[i]         # (1,1) - scalar = (1,1)
        
        total_loss += float(error[0][0] ** 2)
        
        #calculate mse
        dy_preds = [np.zeros((1, 1))] * 5 + [error] #! its why we only use last stm set to make a prediciton and other 5 only effect backward_sequence
        
        # backward
        accumulated_grads = cell.backward_sequence(dy_preds=dy_preds, stm_outputs=window_stm_outputs, caches=window_caches,sigma=SİGMA)
        cell.update_weights(grads=accumulated_grads, learning_rate=0.01)
        
    if (step+1) % 100 == 0:
        print(f"step {step} Loss: {total_loss / len(window_x_seq):.4f}")