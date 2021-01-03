from utils_tf2 import* 
import numpy as np
import tensorflow as tf
import pickle

'''
This is a Neural Network to predict the resonant frequency, The structure is:
Train/Test split = 0.85
hidden layers = 5
neurons in each layer = 6
learning rate = 0.0085
number of epochs = 3600000
'''

# Specify Inputs, Outputs, Train/Test split, and Datafile (arguments of the load_dataset function)
data = "datasets/Data_extended_all.csv"
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width']
list_of_outputs = ['resonance']
W = [9, 9, 9, 9, 9, 1]
activations = ["relu", "relu", "relu", "relu", "relu"]
frac = 0.85
norm_mean = "normalizations/resonance_extended_all_mean_value.pkl"
norm_std = "normalizations/resonance_extended_all_std_value.pkl"
param = "parameters/resonance_extended_all_parameters.pkl"

# Obtain Train and Test sets inputs and outputs and normalize inputs
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(data, frac, list_of_inputs, list_of_outputs, consistent = True, seed = 100)
train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x, name_mean = norm_mean, name_std = norm_std)

# Structure and build the NN by defining the number of layers, neurons, and activations
parameters = model(train_set_x, train_set_y, test_set_x, test_set_y, W, activations,learning_rate = 0.0055, num_epochs = 200000, b1 = 0.95)

# Specify sample to test and predict
sample_num = 5
sample = np.reshape(test_set_x[:, sample_num], [test_set_x.shape[0], 1])
print("the sample is: ", sample)
print(sample.shape)

# Predict 
my_prediction = predict(sample, parameters, activations)
print("The prediction is", my_prediction)
print("The true value is: ",test_set_y[:, sample_num] )

# Save parameters as binary into .pkl file
f = open(param, 'wb')
pickle.dump(parameters, f)
f.close()
print("Parameters have been saved")

