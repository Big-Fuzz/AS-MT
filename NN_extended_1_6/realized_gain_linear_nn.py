from utils_tf2 import* 
import numpy as np
import tensorflow as tf
import pickle
import math

'''
This is an NN for prediction of realized gain. The structure is:

Train/Test split = 0.85
hidden layers = 4 (tanh, tanh, tanh, relu)
neurons in each layer = 7, 7, 7, 7
learning rate = 0.001
number of epochs = 500000
'''

# Specify Inputs, Outputs, Train/Test split, and Datafile (arguments of the load_dataset function)
data = "datasets/Data_1_6.csv"
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth', 'feed_gap_width', 'resonance']
list_of_outputs = ['realized_gain']
W = [7, 7, 7, 7, 1]
activations = ["tanh", "tanh", "tanh", "relu"]
frac = 0.85
norm_mean = "normalizations/realized_gain_extended_1_6_linear_mean_value.pkl"
norm_std = "normalizations/realized_gain_extended_1_6_linear_std_value.pkl"
param = 'parameters/realized_gain_extended_1_6_linear_parameters.pkl'

# Obtain Train and Test sets inputs and outputs
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(data, frac, list_of_inputs, list_of_outputs, consistent = True, seed = 100)

train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x, name_mean = norm_mean, name_std = norm_std)
train_set_y = 10 ** (train_set_y / 10)
test_set_y = 10 ** (test_set_y / 10)

# Structure and build the NN by defining the number of layers, neurons, and activations
parameters = model(train_set_x, train_set_y, test_set_x, test_set_y, W, activations,learning_rate = 0.001, num_epochs = 500000, print_errors = True, gpu = True)

# Specify sample to test and predict
sample_num = 5
sample = np.reshape(test_set_x[:, sample_num], [test_set_x.shape[0], 1])
print("the test sample is: ", sample)
print(sample.shape)

# Predict 
my_prediction = predict(sample, parameters, activations)
print("The prediction is", my_prediction)
print("The true value is: ",test_set_y[:, sample_num])

print("The prediction in dB is", 10 * math.log10(my_prediction))
print("The true value in dB is: ",10 * math.log10(test_set_y[:, sample_num]))


# Save parameters as binary into .pkl file
f = open(param, 'wb')
pickle.dump(parameters, f)
f.close()
print("parameters/Parameters have been saved")



