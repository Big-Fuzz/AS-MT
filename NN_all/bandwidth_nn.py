from utils_tf2 import* 
import numpy as np
import tensorflow as tf
import pickle

'''
This is a Neural Network to predict the resonant frequency, The structure is:
Train/Test split = 0.8
hidden layers = 5
neurons in each layer = 25, 20, 8, 4, 4
learning rate = 0.0025
number of epochs = 3600000
'''

# Specify Inputs, Outputs, Train/Test split, and Datafile (arguments of the load_dataset function)
data = "datasets/Data_all_with_bandwidth.csv"
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width','resonance']
list_of_outputs = ['bandwidth']
W = [25, 24, 23, 22, 1]
activations = ["tanh", "tanh", "tanh", "tanh"]
frac = 0.8
norm_mean = "normalizations/bandwidth_all_mean_value.pkl"
norm_std = "normalizations/bandwidth_all_std_value.pkl"
param = "parameters/bandwidth_all_parameters.pkl"
drop_layers = []
drop_rates = []
# Obtain Train and Test sets inputs and outputs and normalize inputs
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(data, frac, list_of_inputs, list_of_outputs)#, consistent = True, seed = 100)
train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x, name_mean = norm_mean, name_std = norm_std)

# Structure and build the NN by defining the number of layers, neurons, and activations
parameters = model(train_set_x, train_set_y, test_set_x, test_set_y, W, activations, learning_rate = 0.0033, num_epochs = 120000, dropout_layers = drop_layers, dropout_rates = drop_rates, b1 = 0.97)

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

