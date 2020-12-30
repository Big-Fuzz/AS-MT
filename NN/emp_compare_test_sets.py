from utils_tf2 import * 
import numpy as np
import tensorflow as tf
import math
import pickle

'''
This program compares the performance in predicting/calculations for the resonant frequency of:
   1.) The trained standard NN
   2.) The empirical Formula
with the true value
'''

#'''
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()                # to be able to rerun the model without overwriting tf variables

a_file = open("resonance_parameters.pkl", "rb")

parameters = pickle.load(a_file)
activations = ["relu", "relu", "relu", "relu", "relu"]

print(parameters)
a_file.close()

list_of_inputs = ['permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width']
list_of_outputs = ['resonance']
frac = 0.85
data = "datasets/Data_with_header.csv"

train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(data, frac, list_of_inputs, list_of_outputs, consistent = True, seed = 100)
test_set_x_unnormalized = test_set_x
train_set_x_unnormalized = train_set_x
train_set_y = train_set_y / 1000000000
test_set_y = test_set_y / 1000000000
train_set_x, test_set_x = normalize_inputs(train_set_x, test_set_x)

(n_x, m) = train_set_x.shape                          # (n_x: input size, m : number of examples in the train set)
n_y = train_set_y.shape[0]                            # n_y : output size
    
# Create Placeholders of shape (n_x, n_y)
X, Y = create_placeholders(n_x, n_y)

# Forward propagation: Build the forward propagation in the tensorflow graph
Z3 = forward_propagation(X, parameters, activations)

# Initialize all the variables
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    
    # Run the initialization
    sess.run(init)
    
    accuracy = tf.reduce_mean(1 - tf.math.divide(tf.abs(Z3 - Y),tf.abs(Y)))
    train_acc = accuracy.eval({X: train_set_x, Y: train_set_y})
    test_acc = accuracy.eval({X: test_set_x, Y: test_set_y})
    #print("Train Accuracy:", accuracy.eval({X: train_set_x, Y: train_set_y}))
    print("Test Accuracy:", accuracy.eval({X: test_set_x, Y: test_set_y}))
   
    train_samples = train_set_x.shape[1]
    test_samples = test_set_x.shape[1]
    all_samples = train_samples + test_samples
    overall_acc = ((train_samples * train_acc) + ( test_samples * test_acc )) / all_samples   
    #print("Overall Accuracy" , overall_acc) 
    test_samples = test_set_x.shape[1]
    err_sum = 0
    for i in range(0, test_samples):
        sample_num = i
        tru_val = test_set_y[:, sample_num]
        sample_L = test_set_x_unnormalized[:, sample_num][1]
        sample_W = test_set_x_unnormalized[:, sample_num][2]
        sample_h = test_set_x_unnormalized[:, sample_num][3]
        sample_e_r = test_set_x_unnormalized[:, sample_num][0]
        eq = resonance_emp(sample_L, sample_W, sample_h, sample_e_r, mm = True) / 10 **9
        err = (abs(tru_val - eq)) / (abs(tru_val))
        err_sum = err_sum + err
    print ("The average accuracy of the equation: ", 1 - err_sum / test_samples )
     
