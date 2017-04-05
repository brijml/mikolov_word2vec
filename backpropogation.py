import numpy as np


def sigmoid(z):
	return 1/(1+np.exp(-z))

def error_derivative_linear(delta,activation):

	return delta-activation

def previous_delta_term(delta,weight_vector,activation):

	return (np.matrix(weight_vector).T * delta) * (activation * (1-activation))

def do_forward_propogation(x,weights):

	global activation

	activation = np.zeros((r,c),dtype = np.float32)
	activation[0,:] = x
	for i in range(1,c):
		activation[:,i] = sigmoid(activation[:,i-1],weights[:,i])

	return 



def backpropogation(x,target,weights):
	"""Standard implementaion of backpropogation to calculate the error derivatives for a single input value"""

	do_forward_propogation(x,weights)
	
	delta = np.zeros((r,c),dtype = np.float32)
	delta[c,:] = target - activation[c,:]
	for i in range(1,c):
		delta[:,c-i] = previous_delta_term(delta[:,c-i+1],weight_vector[:,c-i],activation[:,c-i])

	for j in range(1,c):
		error_derivatives = activation[:,c] * delta[:,c+1]


	return error_derivatives


if __name__ == '__main__':
	#for all the input do backpropogation 
	global r,c
	r,c = weights.shape
	error_derivatives = np.zeros((r,c),dtype = np.float32)
	weights = initialise_weights()

	for x,target in all_inputs:
		error_derivates = backpropogation(x,target,weights,error_derivatives)

