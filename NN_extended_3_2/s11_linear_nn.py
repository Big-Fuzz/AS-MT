from utils_tf2 import*
import numpy as np
import tensorflow as tf
import pickle

'''
#This is an NN for prediction of the S11 parameters value. The structure is:

Train/Test split = 0.85
hidden layers = 5 (tanh, tanh, tanh, tanh, relu)
neurons in each layer = 7 7 7 7 8
learning rate = 0.0005
number of epochs = 100000 early stopping for regularizing
'''

# Specify Inputs, Outputs, Train/Test split, and Datafile (arguments of the load_dataset function)
data = "datasets/Data_3_2.csv"
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width', 'resonance']
list_of_outputs = ['s11']
W = [7, 7, 7, 7, 8, 1]
activations = ["tanh", "tanh", "tanh", "tanh", "relu"]
frac = 0.85
norm_mean = "normalizations/s11_extended_3_2_linear_mean_value.pkl"
norm_std = "normalizations/s11_extended_3_2_linear_std_value.pkl"
param = 'parameters/S11_extended_3_2_parameters_linear.pkl'

# Obtain Train and Test sets inputs and outputs
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(data, frac, list_of_inputs, list_of_outputs, consistent = True, seed = 100)
train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x, name_mean = norm_mean, name_std = norm_std)
train_set_y = 10 ** (train_set_y / 20) 
test_set_y = 10 ** (test_set_y / 20) 

# Structure and build the NN by defining the number of layers, neurons, and activations
parameters = model(train_set_x, train_set_y, test_set_x, test_set_y, W, activations,learning_rate = 0.0005, num_epochs = 100000)

# Specify sample to test and predict
sample_num = 5
sample = np.reshape(test_set_x[:, sample_num], [test_set_x.shape[0], 1])
print("the sample is: ", sample)
print(sample.shape)

# Predict 
my_prediction = predict(sample, parameters, activations)
print("The prediction is", my_prediction)
print("The true value is: ",test_set_y[:, sample_num])

# Save parameters as binary into .pkl file
f = open(param, 'wb')
pickle.dump(parameters, f)
f.close()
print("Parameters have been saved")
