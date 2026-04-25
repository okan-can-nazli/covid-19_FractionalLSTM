from CaputoLstm import LSTMCell
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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

########################################### NORMALIZATION ###########################################
turkey["New_cases"] = np.log1p(turkey["New_cases"]) # log(x+1) for x = 0 protection
# Before:   min=0,  max=406,321,  std=28,954
# After:    min=0.69,  max=11.6,  std=1.36


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


########################################### TRAIN ###########################################




#! CONSTANTS
seed_num = 42
np.random.seed(seed_num)
SIGMA = 1
STM_SIZE = 16
EPOCHS = 5000
lr = 0.01


print("Starting to train....")
print(f"===== Constants =====\nNumpy Random Seed: {seed_num if seed_num is not None else 'Unset (Random)'}\nSTM Size: {STM_SIZE}\nLearning Rate: {lr}\nSIGMA:{SIGMA}")

# init cell
cell = LSTMCell(input_size = 1,stm_size = STM_SIZE,output_size = 1) #! input_size means multivariate data!!!


train_dates = train_data["Date_reported"].iloc[6:].reset_index(drop=True) # for matplotlib x axis representation

for step in range(EPOCHS):
    
    total_loss = 0
    train_predictions = []
    train_expected_values = []
    
    window_outputs = []
    dy_preds = []
    window_caches = []
    

    
    # forward
    for i, window in enumerate(window_x_seq): 
        window_stm_outputs, window_caches = cell.forward_sequence(x_sequence = window, stm_init = np.zeros((STM_SIZE, 1)), ltm_init = np.zeros((STM_SIZE, 1)))
        
        
        prediction = cell.predict(window_stm_outputs[-1])  # (16,1) → (1,1)
        error = prediction - window_y_seq[i]         # (1,1) - scalar = (1,1)
        
        total_loss += float(error[0][0] ** 2) # error in np 2d array format
        
        train_expected_values.append(window_y_seq[i]) # for matplotlib
        train_predictions.append(float(prediction[0][0]))
        
        #calculate mse
        dy_preds = [np.zeros((1, 1))] * 5 + [error] #! its why we only use last stm set to make a prediciton and other 5 only effect backward_sequence
        
        # backward
        accumulated_grads = cell.backward_sequence(dy_preds=dy_preds, stm_outputs=window_stm_outputs, caches=window_caches,sigma=SIGMA)
        cell.update_weights(grads=accumulated_grads, learning_rate=lr)
        
    if (step+1) % 100 == 0:
        print(f"step {step+1} Loss: {total_loss / len(window_x_seq):.8f}")
        

# matplotlib for training section
train_predictions_real = np.expm1(train_predictions) # convert New_case values back
train_expected_real = np.expm1(train_expected_values) # convert New_case values back

plt.title(f"Train Data Sigma: {SIGMA}")
plt.plot(train_dates, train_expected_real, label="actual")
plt.plot(train_dates, train_predictions_real, label="predicted")
plt.legend()
plt.savefig(f"TRAİN_sigma_{SIGMA}.png")
plt.close()
        
########################################### TESTING ###########################################
print("Starting to testing...")

test_x_seq = []
test_y_seq = []

for i in range(len(test_data)-6):
    window = test_data["New_cases"].iloc[i:i+7]
    x = window.iloc[:6]
    y = window.iloc[6]
    test_x_seq.append(x)
    test_y_seq.append(y)
    
# our cell expects np arrays
test_x_seq = np.array(test_x_seq)
test_y_seq = np.array(test_y_seq)


test_loss = 0

test_dates = test_data["Date_reported"].iloc[6:].reset_index(drop=True) # for matplotlib x axis representation

test_predictions = []
test_expected_values = []

# forward
for i, window in enumerate(test_x_seq): 
    window_stm_outputs, window_caches = cell.forward_sequence(x_sequence = window, stm_init = np.zeros((STM_SIZE, 1)), ltm_init = np.zeros((STM_SIZE, 1)))    
    
    prediction = cell.predict(window_stm_outputs[-1])  # (16,1) → (1,1)
    error = prediction - test_y_seq[i]         # (1,1) - scalar = (1,1)
    
    test_expected_values.append(test_y_seq[i]) # for matplotlib
    test_predictions.append(float(prediction[0][0]))
    test_loss += float(error[0][0] ** 2) # error in np 2d array format

print(f"Test Loss: {test_loss / len(test_x_seq):.8f}")


# matplotlib for testing section
test_predictions_real = np.expm1(test_predictions) # convert New_case values back
test_expected_real = np.expm1(test_expected_values) # convert New_case values back

plt.title(f"Test Data Sigma: {SIGMA}")
plt.plot(test_dates, test_expected_real, label="actual")
plt.plot(test_dates, test_predictions_real, label="predicted")
plt.legend()
plt.savefig(f"TEST_sigma_{SIGMA}.png")
plt.close()