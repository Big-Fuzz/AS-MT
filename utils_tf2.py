import h5py
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import math
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
#import transformers
import pickle


def load_dataset(data, frac, list_of_inputs, list_of_outputs, consistent = False, seed = 4):
    """
    Loads the dataset and splits into into train/test-set with the given proportion split
    and returns the inputs and outputs of each set

    Arguments:
    data -- string with name of datafile (datafile of shape (samples, features))
    frac -- the split for the training set
    list_of_inputs -- list with strings of input columns names
    list_of_outputs -- list with strings of output columns names
    consistent -- False which means random split of sets
    seed -- value of random seed, works only if consistent is set to True

    Returns:
    train_set_x -- inputs of training set of shape (inputs, number of samples)
    train_set_y -- outputs of training set (labels) of shape (output, number of samples)
    test_set_x -- inputs of test set of shape (inputs, number of samples)
    test_set_y -- outputs of test set (labels) of shape (output, number of samples)
    """
    
    if consistent == True:
        df = pd.read_csv(data)
        print("Dataset shape:", df.shape)
        rng = RandomState(seed)
        
    else:
        df = pd.read_csv(data)
        print("Dataset shape:", df.shape)
        rng = RandomState()
	
    train_set = df.sample(frac=frac, random_state=rng)
    test_set = df.loc[~df.index.isin(train_set.index)]

    train_set_x = np.array(train_set[list_of_inputs][:]) 
    train_set_y = np.array(train_set[list_of_outputs][:])

    test_set_x = np.array(test_set[list_of_inputs][:]) 
    test_set_y = np.array(test_set[list_of_outputs][:])

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
	
    return train_set_x.T, train_set_y, test_set_x.T, test_set_y

def separate_dataset(data, list_of_inputs, list_of_outputs):
	"""
	Seperates the dataset into inputs and outputs
	
	Arguments:
	data -- string with name of datafile
	list_of_inputs -- list with strings of input columns names
	list_of_outputs -- list with strings of output columns names

	Returns:
	x -- inputs
	y -- outputs(labels)
	"""

	df = data
	print("Dataset shape:", df.shape)

	x = np.array(df[list_of_inputs][:]) 
	y = np.array(df[list_of_outputs][:])
	
	y = y.reshape((1, y.shape[0]))
	
	return x.T, y


def normalize_inputs(train, test, name_mean = "normalizations/mean_value.pkl", name_std = "normalizations/std_value.pkl"):
    """
    Arguments:
    train -- the training set inputs
    test -- the test set inputs

    Returns:
    train_norm -- the normalized training set inputs
    test_norm -- the normalized test set inputs
    name_mean -- name of the file where the mean value is stored
    name_std -- name of the file where the standard deviation value is stored
    """

    mean = []
    std = []
    for i in range(train.shape[0]):
        if np.std(train[i,:]) == 0:
            mean.append([0])
            std.append([1])
        else:
            mean.append([np.mean(train[i,:])])
            std.append([np.std(train[i,:])])
    train_norm = (train - np.array(mean)) / np.array(std)
    test_norm = (test - np.array(mean)) / np.array(std)
    
    f = open(name_mean, 'wb')
    pickle.dump(mean, f)
    f.close()
    
    f = open(name_std, 'wb')
    pickle.dump(std, f)
    f.close()

    return train_norm, test_norm
    
def normalize_with_existing(train, test, name_mean, name_std ):
    """
    Arguments:
    train -- the training set inputs
    test -- the test set inputs
    name_mean -- name of the file where the mean value is stored
    name_std -- name of the file where the standard deviation value is stored

    Returns:
    train_norm -- the normalized training set inputs
    test_norm -- the normalized test set inputs
    """
    
    a_file = open(name_mean, "rb")
    mean = pickle.load(a_file)

    a_file = open(name_std, "rb")
    std = pickle.load(a_file)

    train_norm = (train - np.array(mean)) / np.array(std)
    test_norm = (test - np.array(mean)) / np.array(std)

    return train_norm, test_norm
    
def normalize_with_existing_wholedf(df, name_mean, name_std ):
    """
    Arguments:
    train -- the training set inputs
    test -- the test set inputs
    name_mean -- name of the file where the mean value is stored
    name_std -- name of the file where the standard deviation value is stored

    Returns:
    train_norm -- the normalized training set inputs
    test_norm -- the normalized test set inputs
    """
    
    a_file = open(name_mean, "rb")
    mean = pickle.load(a_file)

    a_file = open(name_std, "rb")
    std = pickle.load(a_file)

    df_norm = (df - np.array(mean)) / np.array(std)

    return df_norm
     
def create_placeholders(n_x, n_y, m = 0):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of input
    n_y -- scalar, size of output 
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    m -- number of samples if they are known, else m = 0 by default which sets the outputs tensor shape to (num of outputs, None)
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    
    if m == 0:
    	X = tf.compat.v1.placeholder("float", [n_x, None])
    	Y = tf.compat.v1.placeholder("float", [n_y, None])
    else:
        X = tf.compat.v1.placeholder("float", [n_x, m])
        Y = tf.compat.v1.placeholder("float", [n_y, m])


    return X, Y


def initialize_parameters(W, n_x):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [W[0], n_x]
                        b1 : [b[0], 1]
                        W2 : [W[1], W[0]]
                        b2 : [b[1], 1]
                        ...
                        ...
    Arguments:
	W -- list with number of neurons per layer, the layer being the index of the list
	n_x -- number of inputs

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    #tf.set_random_seed(1)       # so that your "random" numbers are consistent
    parameters = {}
    parameters["W1"] = tf.compat.v1.get_variable("W1", [W[0],n_x], initializer = tf.keras.initializers.GlorotNormal()) 
    #parameters["W1"] = tf.compat.v1.get_variable("W1", [W[0],n_x], initializer = tf.random_normal_initializer()) 
    parameters["b1"] = tf.compat.v1.get_variable("b1", [W[0],1], initializer = tf.zeros_initializer()) 

    for i in range(1,len(W)):
        parameters["W" + str(i+1)] = tf.compat.v1.get_variable("W" + str(i+1), [W[i],W[i-1]], initializer = tf.keras.initializers.GlorotNormal()) 
        #parameters["W" + str(i+1)] = tf.compat.v1.get_variable("W" + str(i+1), [W[i],W[i-1]], initializer = tf.random_normal_initializer()) 
        parameters["b" + str(i+1)] = tf.compat.v1.get_variable("b" + str(i+1), [W[i],1], initializer = tf.zeros_initializer()) 
     
    return parameters


def forward_propagation(X, parameters, activations, dropout_layers = [], dropout_rates= []):
    """
    Implements the forward propagation for the model: LINEAR -> activation -> LINEAR -> activation .... -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"....
                  the shapes are given in initialize_parameters
    activations -- a list containing the activations for each hidden layer in order    
    dropout_layers -- list whose elements indicate the number of the layer on which dropout should be applied. Example [1,3,4]
    dropout_rates -- list whose elements define the rate of the dropout that is applied to the layers of same index value in dropout_layers. Example [0.3, 0.5, 0.7]]

    Returns:
    Z_final -- the output of the last LINEAR unit (number of outputs, number of samples (m))
    """
    
    if len(activations) == (len(parameters)/2 - 1):
        
        if len(activations) >= 2:
            equations = {}
            Z1 = tf.add(tf.matmul(parameters['W1'], X), parameters['b1'])         #Z1 = np.dot(parameters['W1'], X) + parameters['b1']

            if activations[0] == "relu":
                #print("Relu activated")
                A1 = tf.nn.relu(Z1)    
            elif activations[0] == "sigmoid":
                #print("Sigmoid activated")
                A1 = tf.nn.sigmoid(Z1)
            elif activations[0] == "tanh":
                #print("Tanh activated")
                A1 = tf.nn.tanh(Z1)
            elif activations[0] == "none":
                #print("No Activation")
                A1 = Z1     
            
            if 1 in dropout_layers:
                tf.nn.dropout(A1, dropout_rates[dropout_layers.index(1)])
                print("applied dropout on layer 1 with rate of", dropout_rates[dropout_layers.index(1)])
                                         
            equations["Z1"] = Z1 
            equations["A1"] = A1 

            for i in range(2, (len(activations) + 1)):
                x = "Z" + str(i)
                y = "A" + str(i)   
                equations["Z" + str(i)] = tf.add(tf.matmul(parameters['W' + str(i)], equations['A' + str(i-1)]), parameters['b' + str(i)])

                if activations[i-1] == "relu":
                    #print("Relu activated.")
                    equations["A" + str(i)] = tf.nn.relu(equations["Z" + str(i)])
                elif activations[i-1] == "sigmoid":
                    #print("Sigmoid activated.")
                    equations["A" + str(i)] = tf.nn.sigmoid(equations["Z" + str(i)])
                elif activations[i-1] == "tanh":
                    #print("Tanh activated.")
                    equations["A" + str(i)] = tf.nn.tanh(equations["Z" + str(i)])
                elif activations[i-1] == "none":
                    #print("No Activation")
                    equations["A" + str(i)] = equations["Z" + str(i)]
                  
                if i in dropout_layers:
                    tf.nn.dropout(equations["A" + str(i)], dropout_rates[dropout_layers.index(i)])
                    print("applied dropout on layer", i, "with rate of", dropout_rates[dropout_layers.index(i)])
                	
            Z_final = tf.add(tf.matmul(parameters['W' + str(i+1)], equations["A" + str(i)]), parameters['b' + str(i+1)])
            #Z_final = tf.nn.relu(Z_final)
            return Z_final
  
        elif len(activations) == 1: 
            print("only one activation")
            equations = {}
            Z1 = tf.add(tf.matmul(parameters['W1'], X), parameters['b1'])         #Z1 = np.dot(parameters['W1'], X) + parameters['b1']

            if activations[0] == "relu":
                #print("Relu activated")
                A1 = tf.nn.relu(Z1)  
            if activations[0] == "sigmoid":
                #print("Sigmoid activated")
                A1 = tf.nn.sigmoid(Z1)    
            if activations[0] == "tanh":
                #print("Tanh activated")
                A1 = tf.nn.tanh(Z1)  
            if activations[0] == "none":
                #print("No Activation")
                A1 = tf.nn.tanh(Z1)  
                
            if 1 in dropout_layers:                 
                tf.nn.dropout(A1, dropout_rates[dropout_layers.index(1)])
                print("applied dropout on layer 1 with rate of", dropout_rates[dropout_layers.index(1)])   
  
            equations["Z1"] = Z1 
            equations["A1"] = A1    

        Z_final = tf.add(tf.matmul(parameters['W' + str(2)], equations["A" + str(1)]), parameters['b' + str(2)])
        #Z_final = tf.nn.relu(Z_final)
        return Z_final

    else:
        return 


def compute_cost(z3, Y):

    """
    Computes the cost
    
    Arguments:
    z3 -- output of forward propagation (output of the last LINEAR unit), of shape (output of before last layer, number of examples)
    Y -- "true" labels vector placeholder, same shape as z3
    
    Returns:
    cost - Tensor of the cost function
    """  
    
    predictions = z3
    labels = Y

    if predictions.shape[0] == 2:
        var0 = tf.slice(predictions, [0, 0], [1, predictions.shape[1]])
        var1 = tf.slice(predictions, [1, 0], [1, predictions.shape[1]])
        print(var0, var1)
        predictions = var0 * var1 
        cost = tf.keras.losses.mse(labels, predictions)
    else:
        cost = tf.keras.losses.mse(labels, predictions)

    return cost


def compute_error(Z3_train, Z3_test, Y_train, Y_test):

    """
    Computes the error or accuracy inverse
    
    Arguments:
    Z3_train -- output of forward propagation on training set (output of the last LINEAR unit), of shape (1, number of examples)
    Z3_test -- output of forward propagation on test set (output of the last LINEAR unit), of shape (1, number of examples)
    Y_train -- "true" labels of training set vector placeholder, same shape as Z3_train
    Y_test -- "true" labels of test set vector placeholder, same shape as Z3_test
    
    Returns:
    error_train - Tensor of error of training set
    error_test - Tensor of error of test set
    """  
    if Z3_train.shape[0] == 2:
        var0 = tf.slice(Z3_train, [0, 0], [1, Z3_train.shape[1]])
        var1 = tf.slice(Z3_train, [1, 0], [1, Z3_train.shape[1]])   
        Z =  var0 * var1 
        error_train = tf.reduce_mean(tf.math.divide(tf.abs(Z - Y_train),tf.abs(Y_train)))       

        var3 = tf.slice(Z3_test, [0, 0], [1, Z3_test.shape[1]])
        var4 = tf.slice(Z3_test, [1, 0], [1, Z3_test.shape[1]])     
        Z1 = var3 * var4   
        error_test = tf.reduce_mean(tf.math.divide(tf.abs(Z1 - Y_test),tf.abs(Y_test)))    
            
    else:
        error_train = tf.reduce_mean(tf.math.divide(tf.abs(Z3_train - Y_train),tf.abs(Y_train)))
        error_test = tf.reduce_mean(tf.math.divide(tf.abs(Z3_test - Y_test),tf.abs(Y_test)))
    
    return error_train, error_test


def model(X_train, Y_train, X_test, Y_test, W, activations, learning_rate = 0.0001,
          num_epochs = 1500, print_cost = True, print_errors = True, show_plots = True, gpu = False, b1 = 0.9, b2 = 0.999, dropout_layers = [], dropout_rates= [], get_errors = False):
    """
    Implements a "len(activations)"-layered tensorflow neural network
    
    Arguments:
    X_train -- training set, of shape 
    Y_train -- test set, of shape 
    X_test -- training set, of shape 
    Y_test -- test set, of shape 
    W -- list where each value is number of neurons and index of that value indicates of which hidden layer but last value refers to output of output layer
    activations -- list of activations for each hidden layer, where avlue of index refers to which hidden layer
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs
    print_errors -- True to print the errors of Train and Test set every 100 epochs
    show_plots -- True (default), shows the plots of cost and test/train error over iterations
    gpu -- True to enable usage of gpu processing
    b1 -- Momentum variable beta1 for adam optimizer
    b2 -- RMS prop variable beta2 for adam optimizer
    dropout_layers -- list whose elements indicate the number of the layer on which dropout should be applied. Example [1,3,4]
    dropout_rates -- list whose elements define the rate of the dropout that is applied to the layers of same index value in dropout_layers. Example [0.3, 0.5, 0.7]]
    get_test_error -- False, if True returns the test set error
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used with the predict function.
    """
    
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()                # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.set_random_seed(1)                   # to keep consistent results
    seed = 0                                          # set initial seed value
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    (n_x_te, m_te) = X_test.shape                     # (n_x: input size, m : number of examples in the train set)
    n_y_te = Y_test.shape[0]                          # n_y : output size
    costs = []                                        # To keep track of the cost
    errors_train = []                                 # To keep track of training errors   
    errors_test = []                                  # To keep track of test errors   
    dev = '/cpu:0'                                    # Default device to use for processeing: CPU
    
    if gpu == True:
        dev = '/gpu:0' 
       
    # Run the following on CPU
    with tf.device('/cpu:0'):
    
        # Create Placeholders of shape (n_x, n_y)
        X, Y = create_placeholders(n_x, n_y, m = m)
        
        # Create Placeholders for test set of shape (n_x, n_y)
        X_te, Y_te = create_placeholders(n_x_te, n_y_te, m = m_te)

        # Initialize parameters
        parameters = initialize_parameters(W, n_x)
       
        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = forward_propagation(X, parameters, activations, dropout_layers, dropout_rates)
        
        # Forward propagation for test set: Build the forward propagation in the tensorflow graph
        Z3_te = forward_propagation(X_te, parameters, activations)

        # Cost function: Add cost function to tensorflow graph
        cost = compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate, beta1 = b1, beta2 = b2).minimize(cost)
      
        # Calculate error
        error, error_te = compute_error(Z3, Z3_te, Y, Y_te)
        
        # Initialize all the variables
        init = tf.compat.v1.global_variables_initializer()
        
    start = time.time()
    
    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Choose which processor to run the following code on
        with tf.device(dev):
        
            # Do the training loop
            for epoch in range(num_epochs):

                seed = seed + 1

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch or fullbatch for (X,Y).
                _ , cost_iter, err, err_te = sess.run([optimizer, cost, error, error_te], feed_dict={X: X_train, Y: Y_train, X_te: X_test, Y_te: Y_test})              
                
                # Print the cost every 100 epochs
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, cost_iter))
                if print_cost == True and epoch % 100 == 0:
                    costs.append(cost_iter)
                
                # Calculate and print the training error for train and test every 100 epochs
                if print_errors == True and epoch % 100 == 0:
                       
                    errors_train.append(err)
                    errors_test.append(err_te)
                    print("Training error: ", err)
                    print("Test error: ",err_te)
                    
        end = time.time()
        print("Device Driver execution runtime: ", (end-start))
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        if show_plots == True:
            plt.show()

        # plot the errors
        plt.plot(np.squeeze(errors_train))
        plt.plot(np.squeeze(errors_test))
        plt.legend(['Training errors','Test errors'])
        plt.ylabel('errors')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        if show_plots == True:
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        print("Train Accuracy:", 1- err)
        print("Test Accuracy:", 1 - err_te)
        print("Train Error:", err)
        print("Test Error:", err_te)        
        
        if get_errors == True:
            return parameters, err, err_te
        else:
            return parameters
        

def modelW(X_train, Y_train, X_test, Y_test, W, activations, learning_rate = 0.0001,
          num_epochs = 1500, print_cost = True, print_errors = True, gpu = False):
    """
    Implements a "len(activations)"-layered tensorflow neural network
    
    Arguments:
    X_train -- training set, of shape
    Y_train -- test set, of shape 
    X_test -- training set, of shape 
    Y_test -- test set, of shape 
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs
    print_errors -- True to print the errors of Train and Test set every 100 epochs
    gpu -- True to enable usage of gpu processing
    b1 -- Momentum variable beta1 
    b2 -- RMS prop variable beta2
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()                # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.set_random_seed(1)                   # to keep consistent results
    seed = 0                                          # set initial seed value
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    (n_x_te, m_te) = X_test.shape                     # (n_x: input size, m : number of examples in the train set)
    n_y_te = Y_test.shape[0]                          # n_y : output size
    costs = []                                        # To keep track of the cost
    errors_train = []                                 # To keep track of training errors   
    errors_test = []                                  # To keep track of test errors   
    dev = '/cpu:0'                                    # Default device to use for processeing: CPU
    
    if gpu == True:
        dev = '/gpu:0' 
       
    # Run the following on CPU
    with tf.device('/cpu:0'):
    
        # Create Placeholders of shape (n_x, n_y)
        X, Y = create_placeholders(n_x, n_y)
        
        # Create Placeholders for test set of shape (n_x, n_y)
        X_te, Y_te = create_placeholders(n_x_te, n_y_te)

        # Initialize parameters
        parameters = initialize_parameters(W, n_x)
       
        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = forward_propagation(X, parameters, activations)
        
        # Forward propagation for test set: Build the forward propagation in the tensorflow graph
        Z3_te = forward_propagation(X_te, parameters, activations)

        # Cost function: Add cost function to tensorflow graph
        cost = compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        #grad = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(cost)
        vl = tfa.optimizers.AdamW(weight_decay = 0.1, learning_rate = learning_rate).get_weights()
        print("weights", vl)
        optimizer = tfa.optimizers.AdamW(weight_decay = 0.1, learning_rate = learning_rate).minimize(loss = cost, var_list = vl)
        #optimizer = transformers.AdamWeightDecay(weight_decay_rate = 0.1, learning_rate = learning_rate).minimize(cost)
      
        # Calculate error
        error, error_te = compute_error(Z3, Z3_te, Y, Y_te)
        
        # Initialize all the variables
        init = tf.compat.v1.global_variables_initializer()
        
    start = time.time()
    
    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Choose which processor to run the following code on
        with tf.device(dev):
        
            # Do the training loop
            for epoch in range(num_epochs):

                seed = seed + 1

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch or fullbatch for (X,Y).
                _ , cost_iter, err, err_te = sess.run([optimizer, cost, error, error_te], feed_dict={X: X_train, Y: Y_train, X_te: X_test, Y_te: Y_test})              
                
                #_ , cost_iter, err, err_te, gr = sess.run([optimizer, cost, error, error_te, grad[1][0] ], feed_dict={X: X_train, Y: Y_train, X_te: X_test, Y_te: Y_test})
                #print(gr)
                
                # Print the cost every 100 epochs
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, cost_iter))
                if print_cost == True and epoch % 100 == 0:
                    costs.append(cost_iter)
                
                # Calculate and print the training error for train and test every 100 epochs
                if print_errors == True and epoch % 100 == 0:
                       
                    errors_train.append(err)
                    errors_test.append(err_te)
                    print("Training error: ", err)
                    print("Test error: ",err_te)

        end = time.time()
        print("Device Driver execution runtime: ", (end-start))
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # plot the errors
        plt.plot(np.squeeze(errors_train))
        plt.plot(np.squeeze(errors_test))
        plt.legend(['Training errors','Test errors'])
        plt.ylabel('errors')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        accuracy = tf.reduce_mean(1 - tf.math.divide(tf.abs(Z3 - Y),tf.abs(Y)))
        error = 1 - accuracy
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        print("Train Error:", error.eval({X: X_train, Y: Y_train}))
        print("Test Error:", error.eval({X: X_test, Y: Y_test}))
        print("Cost on Test set: " , cost.eval({X: X_test, Y: Y_test}))
        
        return parameters


def predict(X, parameters, activations):

    '''
    Predicts a certain sample by forward propogation with trained parameters

    Arguments:
    X -- sample to be predicted: input dataset placeholder, of shape (input size, 1)
    parameters -- dictionary containing parameters
    activations -- a list containing the activations for each hidden layer in order   

    Returns:
    prediction -- the resulting value of the forward prop (the prediction)
    '''

    params = {}
    for i in range(1, int(len(parameters)/2+1)):
        
        params["W"+ str(i)] = tf.convert_to_tensor(parameters["W"+ str(i)])
        params["b"+ str(i)] = tf.convert_to_tensor(parameters["b"+ str(i)])

 
    x = tf.compat.v1.placeholder("float", [X.shape[0], X.shape[1]])
    
    z_final = forward_propagation(x, params, activations)
    #z_final = tf.argmax(z_final)
    
    sess = tf.compat.v1.Session()
    prediction = sess.run(z_final, feed_dict = {x: X})
        
    return prediction

def resonance_emp(L, W, h, e_r, mm = False):

    '''
    Calculates the resonant frequency of a microstrip patch antenna using the empirical formula according to (Kark,  Antennen und Strahlungsfelder, 2014 pages 431 - 435)
    
    Arguments:
    L -- length of the patch
    W -- width of the patch
    h -- height of the patch
    e_r -- the relative permittivity of the subtrate
    mm -- False boolean for inputs in m (meters)
    
    Returns
    f -- the calculated resonant frequency    
    '''
    if mm == True:
        L = L * 0.001
        W = W * 0.001
        h = h *0.001
    #Givens
    c = 299792458
    u = W / h
    
    # Calculate ab(u, e_r)
    if (u <= 100) and (u >= 0) and ( e_r >= 1) and (e_r <= 128):
        ab = 0.559 + u/570 - 1/(10.3 * e_r)
    
    # Calculate the static effective permittivity
    e_r_eff_0 = (e_r + 1)/2 + (e_r - 1)/2 * (1 + 10/u)**-ab
    
    # Calculalate the Length extension due to stray(fringing) fields
    dL = h / (2 * math.pi) * (u + 0.366)/(u + 0.556) * (0.28 + (e_r + 1)/e_r * (0.274 + np.log(u + 2.518)))
    
    # Calculate the effective Length of Patch
    L_eff = L + 2 * dL
    
    # Calculate the resonant frequency
    f = c / (2 * L_eff * np.sqrt(e_r_eff_0))
    
    return f
    
def Kfold(splits, data):
    '''
    Splits the dataframe into unintersecting splits of train/test sets
    
    Arguments:
    splits -- integer number of splits to splits the dataframe 
    data -- dataframe
    
    Returns:
    sets -- dictionary containing the the test and train sets in the form (test0, train0, .......)
    '''

    sets = {}
    df = pd.read_csv(data)
    indexes = random.sample(range(0, df.shape[0] ), df.shape[0] )
    df = df.iloc[indexes]

    for i in range(splits):
        test_set = df.iloc[i * int(df.shape[0]/splits) : (i+1) * int(df.shape[0]/splits)]
        sets["test" + str(i)] = test_set
        sets["train" + str(i)] = df.loc[~df.index.isin(test_set.index)]
    
    return sets  
    
def log10(x):
  
  numerator = tf.compat.v1.log(x)
  denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
  
  return numerator / denominator
  
 
    
