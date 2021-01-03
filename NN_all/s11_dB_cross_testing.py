from utils_tf2 import*
import numpy as np
import tensorflow as tf
import pickle

'''
This is a program for cross testing several runs of trainings and averaging their test score

splits = 5 -----> Train/Test split = 0.80
Kfold_epochs = 4
hidden layers = 5 (tanh, tanh, tanh, tanh, relu)
neurons in each layer = 7, 7, 7, 7, 8
learning rate = 0.0005
number of epochs = 200000
'''


# Specify Inputs, Outputs, Train/Test split, and Datafile (arguments of the load_dataset function)
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width', 'resonance']
list_of_outputs = ['s11']
W = [7, 7, 7, 7, 8, 1]
activations = ["tanh", "tanh", "tanh", "tanh", "relu"]
test_errors = []
train_errors = []
means_of_test_errors = []
means_of_train_errors = []
drop_layers = [5]
drop_rates = [0.2]
splits = 5
data = "datasets/Data_extended_all.csv"
Kfold_epochs = 4

for j in range(Kfold_epochs):
    
    # Obtain Train and Test sets inputs and outputs and normalize inputs
    sets = Kfold(splits, data)
    for i in range(splits):
        test_set = sets["test" + str(i)]
        train_set = sets["train" + str(i)]
        test_set_x, test_set_y = separate_dataset(test_set, list_of_inputs, list_of_outputs)
        train_set_x, train_set_y = separate_dataset(train_set, list_of_inputs, list_of_outputs)

        train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x)

        # Structure and build the NN by defining the number of layers, neurons, and activations        
        parameters, err_train, err_test = model(train_set_x, train_set_y, test_set_x, test_set_y, W, activations, learning_rate = 0.0015, num_epochs = 150000, get_errors = True, show_plots = False, dropout_layers = drop_layers, dropout_rates = drop_rates)
        
        test_errors.append(err_test)
        train_errors.append(err_train)
    mean_test_error = sum(test_errors) / len(test_errors)
    mean_train_error = sum(train_errors) / len(train_errors)
    print(mean_test_error)
    print(mean_train_error)
    means_of_test_errors.append(mean_test_error)
    means_of_train_errors.append(mean_train_error)
        
mean_means_of_test_errors = sum(means_of_test_errors) / len(means_of_test_errors)
mean_means_of_train_errors = sum(means_of_train_errors) / len(means_of_train_errors)
print("Average test error for S11 parameters in dB is:", mean_means_of_test_errors)
print("Average test accuracy for S11 parameters in dB is:", 1 - mean_means_of_test_errors)
print("Average train error for S11 parameters in dB is:", mean_means_of_train_errors)
print("Average train accuracy for S11 parameters in dB is:", 1 - mean_means_of_train_errors)

