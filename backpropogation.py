import numpy as np

def initialise_weights():

	#Good strategy is to initialise small initial weights 

	weights = 0.01 * np.random.randn(D,H)/np.sqrt(D*H)
	return weights

def sigmoid(z,func = "logistic"):
	
	if func == "logistic"
		return 1/(1+np.exp(-z))
	elif func == "tanh"
		return (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))

def error_derivative_linear(delta,activation):
	return delta-activation

def previous_delta_term(delta,weight_vector,activation):
	return (np.matrix(weight_vector).T * delta) * (activation * (1-activation))

def do_forward_propogation(x,weights):
	
	global activation

	activation = np.zeros((D,H),dtype = np.float32)
	activation[0,:] = x
	for i in range(1,c):
		activation[:,i] = sigmoid(activation[:,i-1]*weights[:,i],func = "tanh")

	return 

def backpropogation(x,target,weights):
	"""Standard implementaion of backpropogation to calculate the error derivatives for a single input value
	where the cost function is the squared error measure for all the training samples"""

	do_forward_propogation(x,weights)
	
	delta = np.zeros((D,H),dtype = np.float32)
	delta[c,:] = target - activation[c,:]
	for i in range(1,c):
		delta[:,c-i] = previous_delta_term(delta[:,c-i+1],weight_vector[:,c-i],activation[:,c-i])

	for j in range(1,c):
		capital_delta = activation[:,c] * delta[:,c+1]

	return error_derivatives

def func(x,target,H,alpha):

	global D,H
	
	#D,H = weights.shape #r and c are the rows and columns of the weight matrix where r denotes training samples and c denotes number of hidden layers
	capital_delta = np.zeros((D,H),dtype = np.float32)
	weights = initialise_weights()
	#alpha = 0.001 #learning rate
	regualarise = 0.5 #regularisation


	for x,target in all_inputs:
		capital_delta = backpropogation(x,target,weights,capital_delta)

	error_derivatives = ((1/r) * capital_delta) + (regualarise * weights)

	#use these error derivatives in gradient descent
	#the weight update rule is just error derivatives subtracted from previous weights and scaled using
	#the number of training samples

	weights = weights - error_derivatives

if __name__ == '__main__':
	#for all the input do backpropogation 
	func()