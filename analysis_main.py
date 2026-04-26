from CaputoLstm import LSTMCell
import numpy as np
import matplotlib.pyplot as plt

def create_result_graph(name, dates, expected_real, predictions_real):
# Set a cleaner style and stretch the figure horizontally
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 6), dpi=300)

    plt.title(f"{name} Data Sigma: {SIGMA}", fontsize=14)

    # 1. ACTUAL: Fade the blue line into the background (alpha=0.7) and make it thin
    plt.plot(dates, expected_real, label="actual", color='#1f77b4', alpha=0.7, linewidth=1)

    # 2. PREDICTED: Thicken the orange line and force it to the front (zorder=3)
    plt.plot(dates, predictions_real, label="predicted", color='#ff7f0e', linewidth=2, zorder=3)

    plt.legend(fontsize=12)

    # Prevent the x-axis dates from getting cut off in the saved file
    plt.tight_layout() 
    plt.savefig(f"{name}_sigma_{SIGMA}.png", bbox_inches='tight')
    plt.close()

########################################### TRAIN ###########################################

#! CONSTANTS
seed_num = 42
np.random.seed(seed_num)
#SIGMA = 1
STM_SIZE = 16
EPOCHS = 5000
lr = 0.01

#! Enter lib of the country 
#from Türkiye_data.turkey_set_data import get_data
from Italy_data.italy_set_data import get_data


train_data, test_data, window_x_seq, window_y_seq = get_data()

for SIGMA in [1.0, 0.9, 0.8, 0.7]:
    
    
    print("\nStarting to train....")
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
        
        
        # stm_state = np.zeros((STM_SIZE, 1))
        # ltm_state = np.zeros((STM_SIZE, 1))
        
        
        # forward
        for i, window in enumerate(window_x_seq): 
            window_stm_outputs, window_caches = cell.forward_sequence(x_sequence = window, stm_init = np.zeros((STM_SIZE, 1)), ltm_init = np.zeros((STM_SIZE, 1)))
            
            # stm_state = window_stm_outputs[-1]
            # ltm_state = window_caches[-1][6]  # last cache, ltm_next
            
            prediction = cell.predict(window_stm_outputs[-1])  # (16,1) → (1,1)
            error = prediction - window_y_seq[i]         # (1,1) - scalar = (1,1)
            
            total_loss += float(error[0][0] ** 2) # error in np 2d array format
            
            train_expected_values.append(window_y_seq[i]) # for matplotlib
            train_predictions.append(float(prediction[0][0]))
            
            #calculate mse
            dy_preds = [np.zeros((1, 1))] * 5 + [error] #! its why we only use last stm set to make a prediciton and other 5 only effect backward_sequence
            
            # backward
            accumulated_grads = cell.backward_sequence(dy_preds=dy_preds, stm_outputs=window_stm_outputs, caches=window_caches,sigma=SIGMA)
            cell.update_weights(grads=accumulated_grads, learning_rate = lr)
            
        if (step+1) % 100 == 0:
            print(f"step {step+1} Loss: {total_loss / len(window_x_seq):.8f}")
            

    # matplotlib for training section
    
    train_predictions_real = np.expm1(train_predictions) # convert New_case values back
    train_expected_real = np.expm1(train_expected_values) # convert New_case values back
    
    create_result_graph(name = "Train", dates = train_dates, expected_real = train_expected_real, predictions_real = train_predictions_real)

            
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

    # stm_state = np.zeros((STM_SIZE, 1))
    # ltm_state = np.zeros((STM_SIZE, 1))
    
    # forward
    for i, window in enumerate(test_x_seq): 
        window_stm_outputs, window_caches = cell.forward_sequence(x_sequence = window, stm_init = np.zeros((STM_SIZE, 1)), ltm_init = np.zeros((STM_SIZE, 1)))    
        # stm_state = window_stm_outputs[-1]
        # ltm_state = window_caches[-1][6]
        
        prediction = cell.predict(window_stm_outputs[-1])  # (16,1) → (1,1)
        error = prediction - test_y_seq[i]         # (1,1) - scalar = (1,1)
        
        test_expected_values.append(test_y_seq[i]) # for matplotlib
        test_predictions.append(float(prediction[0][0]))
        test_loss += float(error[0][0] ** 2) # error in np 2d array format

    print(f"Test Loss: {test_loss / len(test_x_seq):.8f}")


    # matplotlib for testing section
    
    test_predictions_real = np.expm1(test_predictions) # convert New_case values back
    test_expected_real = np.expm1(test_expected_values) # convert New_case values back

    create_result_graph(name = "Test", dates = test_dates, expected_real = test_expected_real, predictions_real = test_predictions_real)
