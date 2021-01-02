from utils_tf2 import * 
import numpy as np
import tensorflow as tf
import math
import pickle

'''
This program calculates the average error across a dataset using the empirical formula:
'''

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()                # to be able to rerun the model without overwriting tf variables

data_ = "datasets/Data_1_6.csv"
list_of_inputs = ['permittivity', 'patch_length','patch_width','substrate_height','feed_depth','feed_gap_width']
list_of_outputs = ['resonance']

data = pd.read_csv(data_)
data_x, data_y = separate_dataset(data, list_of_inputs, list_of_outputs)

(n_x, m) = data_x.shape                          # (n_x: input size, m : number of examples in the train set)
n_y = data_y.shape[0]                            # n_y : output size
    
err_sum = 0
for i in range(0, m):
    sample_num = i
    tru_val = data_y[:, sample_num]
    sample_L = data_x[:, sample_num][1]
    sample_W = data_x[:, sample_num][2]
    sample_h = data_x[:, sample_num][3]
    sample_e_r = data_x[:, sample_num][0]
    if data_ == "datasets/Data_with_header.csv":
        eq = resonance_emp(sample_L, sample_W, sample_h, sample_e_r, mm = True)
    else:
        eq = resonance_emp(sample_L, sample_W, sample_h, sample_e_r, mm = True) / 10**9
    err = (abs(tru_val - eq)) / (abs(tru_val))
    err_sum = err_sum + err
print ("The average accuracy of the equation: ", 1 - err_sum[0] / m )    
