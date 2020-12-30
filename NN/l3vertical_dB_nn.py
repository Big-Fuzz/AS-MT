from utils_tf2 import* 
import numpy as np
import tensorflow as tf
import pickle


'''
This is an NN for prediction of the Ludwig vertical parameters value. The structure is:

Train/Test split = 0.85
hidden layers = 3
neurons in each layer =  9, 9, 7 (tanh, tanh, relu)
learning rate = 0.002
number of epochs = 200000
'''

# Specify Inputs, Outputs, Train/Test split, and Datafile (arguments of the load_dataset function)
data = "datasets/Data_with_header.csv"
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width', 'resonance']
list_of_outputs = ['l3vertical']
W = [9, 9, 7, 1]
activations = ["tanh", "tanh", "relu"]
drop_layers = []
drop_rates = []
frac = 0.85
norm_mean = "normalizations/l3v_dB_mean_value.pkl"
norm_std = "normalizations/l3v_dB_std_value.pkl"
param = 'parameters/l3v_dB_parameters.pkl'

# Obtain Train and Test sets inputs and outputs
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(data, frac, list_of_inputs, list_of_outputs, consistent = True, seed = 32)
train_set_x[len(list_of_inputs)-1] = train_set_x[len(list_of_inputs)-1]/(1000000000)
test_set_x[len(list_of_inputs)-1] = test_set_x[len(list_of_inputs)-1]/(1000000000)
train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x, name_mean = norm_mean, name_std = norm_std)

# Structure and build the NN by defining the number of layers, neurons, and activations
parameters = model(train_set_x, train_set_y, test_set_x, test_set_y, W, activations, learning_rate = 0.0002, num_epochs = 200000, dropout_layers = drop_layers, dropout_rates = drop_rates)

# Specify sample to test and predict
sample_num = 5
sample = np.reshape(test_set_x[:, sample_num], [test_set_x.shape[0], 1])
print("the sample is: ", sample)
print(sample.shape)

# Predict 
my_prediction = predict(sample, parameters, activations)
print("The prediction is", my_prediction)
print("The true value is: ",test_set_y[:, sample_num])
print("The linear prediction is: ", 10 ** (my_prediction / 10))
print("The linear true value dB is: ", 10 ** (test_set_y[:, sample_num] / 10))

# Save parameters as binary into .pkl file
f = open(param, 'wb')
pickle.dump(parameters, f)
f.close()
print("Parameters have been saved")

