from utils_tf2 import *
import numpy as np
import tensorflow as tf
import math
import pickle
import pandas as pd



tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()                # to be able to rerun the model without overwriting tf variables

################################################################### Givens:
datas = ["datasets/Data_with_header.csv", "datasets/Data_0_9.csv", "datasets/Data_NN_+_0_9.csv", "datasets/Data_1_6.csv", "datasets/Data_3_2.csv", "datasets/Data_32.csv", "datasets/Data_extended_all.csv", "datasets/Data_all.csv"]
norm_mean = "normalizations/s11_extended_32_dB_mean_value.pkl"
norm_std = "normalizations/s11_extended_32_dB_std_value.pkl"
param = 'parameters/S11_extended_32_parameters_dB.pkl'
activations = ["tanh", "tanh", "tanh", "tanh", "relu"]
list_of_inputs = [ 'permittivity', 'patch_length','patch_width','substrate_height','feed_depth', 'feed_gap_width', 'resonance']
list_of_outputs = ['s11']
dB = True # for dB values or Resonance

#################################################################### Start
for i in datas:
	a_file = open(param, "rb")
	parameters = pickle.load(a_file)
	a_file.close()
	df = pd.read_csv(i)
	data_x, data_y = separate_dataset(df, list_of_inputs, list_of_outputs)

	if i == "datasets/Data_with_header.csv" and list_of_inputs[len(list_of_inputs)-1] == 'resonance':
		data_x[len(list_of_inputs)-1] = data_x[len(list_of_inputs)-1]/(1000000000)
		print("resonance")
		
	if i == "datasets/Data_with_header.csv"and list_of_outputs[0] == 'resonance':
		data_y = data_y / 10**9 
		print("resonance")

	data_x = normalize_with_existing_wholedf(data_x, name_mean = norm_mean, name_std = norm_std)

	if list_of_outputs == ['s11'] and dB == False:
		data_y = 10 ** (data_y / 20) 
		print("dB = False and S11")
		
	elif list_of_outputs != ['s11'] and dB == False:
		data_y = 10 ** (data_y / 10) 
		print("dB = False and not S11")   
		
	(n_x, m) = data_x.shape                          # (n_x: input size, m : number of examples in the train set)
	n_y = data_y.shape[0]                            # n_y : output size
		
	# Create Placeholders of shape (n_x, n_y)
	X, Y = create_placeholders(n_x, n_y)

	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z3 = forward_propagation(X, parameters, activations)

	# Initialize all the variables
	init = tf.compat.v1.global_variables_initializer()

	with tf.compat.v1.Session() as sess:
		
		# Run the initialization
		sess.run(init)
		
		if list_of_outputs == ['s11'] and dB == False:
		    accuracy_linear = tf.reduce_mean(1 - tf.math.divide(tf.abs(Z3 - Y),tf.abs(Y)))
		    acc_linear = accuracy_linear.eval({X: data_x, Y: data_y})
		    
		    accuracy_log = tf.reduce_mean(1 - tf.math.divide(tf.abs(20 * log10(abs(Z3))  - 20 * log10(Y) ), tf.abs(20 * log10(Y) )))
		    acc_log = accuracy_log.eval({X: data_x, Y: data_y})
		    print("dB = False and S11")
		            
		elif list_of_outputs != ['s11'] and dB == False:
		    accuracy_linear = tf.reduce_mean(1 - tf.math.divide(tf.abs(Z3 - Y),tf.abs(Y)))
		    acc_linear = accuracy_linear.eval({X: data_x, Y: data_y})
		    
		    accuracy_log = tf.reduce_mean(1 - tf.math.divide(tf.abs(10 * log10(abs(Z3))  - 10 * log10(Y) ), tf.abs(10 * log10(Y) )))
		    acc_log = accuracy_log.eval({X: data_x, Y: data_y})
		    print("dB = False and not S11")      
		
		elif list_of_outputs == ['s11'] and dB == True:
		    accuracy_log = tf.reduce_mean(1 - tf.math.divide(tf.abs(Z3 - Y),tf.abs(Y)))
		    acc_log = accuracy_log.eval({X: data_x, Y: data_y})    
		    
		    accuracy_linear = tf.reduce_mean(1 - tf.math.divide(tf.abs(10 ** (Z3 / 20)  - 10 ** (Y / 20) ), tf.abs(10 ** (Y / 20) )))
		    acc_linear = accuracy_linear.eval({X: data_x, Y: data_y})      
		    print("dB = True and  S11 accuracy")     
		              
		elif list_of_outputs != ['s11'] and dB == True:
		    accuracy_log = tf.reduce_mean(1 - tf.math.divide(tf.abs(Z3 - Y),tf.abs(Y)))
		    acc_log = accuracy_log.eval({X: data_x, Y: data_y})    
		    
		    accuracy_linear = tf.reduce_mean(1 - tf.math.divide(tf.abs(10 ** (Z3 / 10)  - 10 ** (Y / 10) ), tf.abs(10 ** (Y / 10) )))
		    acc_linear = accuracy_linear.eval({X: data_x, Y: data_y})      
		    print("dB = True and not S11")  
		
		if list_of_outputs == ['resonance']:
		    print("Accuracy: ", acc_log)
		
		else:           
		    print("Linear accuracy: ", acc_linear,i) 
		    print("Log accuracy: ", acc_log,i)


    



 
