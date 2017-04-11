from nltk.corpus import brown
import numpy as np

def initial_C(words,length_of_feature_vector):
	#Take a random probability distribution probably a muti(uni)varitate normal distribution with a 
	#high correlation between the two dimensions or not
	word_count = len(words)
	C = {}
	for word in words:
		C[word] = np.random.randn(length_of_feature_vector)

	return C

def correct(word):

	probability = np.zeros(len(vocabulary))
	index = vocabulary.index(word)
	probability[index] = 1
	return np.matrix(probability).T

def softmax(incoming_vector):
	exponent = np.exp(incoming_vector)
	sum_exponent = np.sum(exponent)
	probability_mass = exponent/sum_exponent

	return np.matrix(probability_mass)

def initialise_weights(arg1,arg2):

	#Good strategy is to initialise small initial weights

	weights = 0.01 * np.random.randn(arg1,arg2)/np.sqrt(arg1*arg2)
	return np.matrix(weights)


def sigmoid(z,func = "logistic"):
	
	if func == "logistic":
		return 1/(1+np.exp(-z))
	elif func == "tanh":
		return (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))

def error_derivative_linear(delta,activation):
	return delta-activation

def previous_delta_term(delta,weights,activation):
	# print 'weights',weights.shape
	# print delta.shape,activation.shape
	activation = np.array(activation)
	a = activation * (1-activation)
	b = weights.T*delta
	# print a.shape,b.shape
	c = a*np.array(b)
	# print c.shape		
	return np.matrix(c)

def do_forward_propogation(x,weights1,weights2):
	
	#This is for the hidden layer
	
	output_hidden = sigmoid(weights1 * x,func = "tanh")

	# print weights2.shape,output_hidden.shape
	#This is for the softmax layer
	# print output_hidden.shape,weights2.shape
	output_softmax = softmax(weights2 * output_hidden)
	# print output_softmax.shape

	return output_hidden,output_softmax

def do_backpropogation(seq,weights1,weights2,output_hidden,output_softmax,target):
	"""Implementaion of backpropogation to calculate the error derivatives for a single input value
	where the cost function is a log likelihood(cross entropy) of the correct word given the sequence of words. 
	The output is a layer of softmax units and the number of units is the size of the vocabulary"""
	
	delta_output = target - output_softmax  #gradient of cost function with respect to the input of the softmax
	# print delta_output.shape,target.shape
	delta_hidden = previous_delta_term(delta_output,weights2,output_hidden)
	delta_C = previous_delta_term(delta_hidden,weights1,seq)

	gradient_hidden_to_softmax =  delta_output * output_hidden.T 
	# print gradient_hidden_to_softmax.shape,weights2.shape
	gradient_input_to_hidden = delta_hidden * seq.T 
	# print gradient_input_to_hidden.shape,weights1.shape  
	gradient_C = np.array(delta_C) * np.array(seq)
	# print gradient_C.shape,seq.shape
	return gradient_hidden_to_softmax,gradient_input_to_hidden,gradient_C


def training(words, learning_rate=0.01, M=30, hidden_units = 40,n = 3,**kwargs):
	
	""" This method takes in the words and returns a learned distribution of the feature vectors of the words and
		the probability function over words.
	
	Parameters
	--------

	param1 : list of words
	
	param2 : float

		Learning rate is defined as the rate at which the model learns, higher the learning rate the model 
		converges faster or probably overshoots. 
	
	param3 : int 
		
		M is the length of vector for distributed representation of each word.
		
	param4 : int

		n = 3 corresponds to a trigram model.

	**kwargs

		Keyword arguments

	Returns
	-------

	C : numpy matrix of size V*M where V is the number of words in the training set

		Learned feature vector of words.

	theta(weights of the neural network) : numpy matrix

		Parameters of the neural network model to be learned
	
	"""
	#global n,M,hidden_units,H

	H = 1 #nuber of hidden layers
	regularise = 0.5 #regularisation
	weights_input_to_hidden = initialise_weights(hidden_units,(n-1)*M)
	weights_hidden_to_softmax = initialise_weights(len(vocabulary),hidden_units)
	# capital_delta = np.zeros((n,H),dtype = np.float32)
	# activation = np.zeros((n,H),dty1pe = np.float32)
	C = initial_C(vocabulary,M)
	number_of_training_samples = len(words)

	cdelta_hidden_to_softmax = np.zeros((len(vocabulary),hidden_units))
	cdelta_input_to_hidden = np.zeros((hidden_units,(n-1)*M))
	cdelta_C = np.matrix(np.zeros((n-1)*M)).T
	for i in range(number_of_training_samples-n-1):
	#generate a sequence of words
		
		input_sequence = np.zeros(((n-1)*M))
		train_words = []
		for j in range(1,n):
			seq_value = words[i+n-j]
			train_words.append(seq_value)
			input_sequence[(j-1)*M:(j*M)] = C[seq_value]
		
		input_sequence = np.matrix(input_sequence).T
		# print input_sequence.shape
		target = correct(words[i+n])
		# capital_delta = backpropogation(x,target,weights,capital_delta)
		output_hidden,output_softmax = do_forward_propogation(input_sequence,weights_input_to_hidden,
																weights_hidden_to_softmax)
	# error_derivatives = ((1/number_of_training_samples) * capital_delta) + (regualarise * weights)
		gradient_hidden_to_softmax,gradient_input_to_hidden,gradient_C = do_backpropogation(input_sequence,
																					weights_input_to_hidden,
																					weights_hidden_to_softmax,
																					output_hidden,
																					output_softmax,target)


		cdelta_hidden_to_softmax += gradient_hidden_to_softmax
		cdelta_input_to_hidden += gradient_input_to_hidden
		# print gradient_C.shape,cdelta_C.shape
		cdelta_C += gradient_C 
		# print weights_input_to_hidden,cdelta_hidden_to_softmax

	weights_hidden_to_softmax = (weights_hidden_to_softmax - learning_rate * cdelta_hidden_to_softmax) + \
									(regularise * weights_hidden_to_softmax)
	weights_input_to_hidden = (weights_input_to_hidden - learning_rate * cdelta_input_to_hidden) + \
								(regularise * weights_input_to_hidden)
	input_sequence = input_sequence - learning_rate * cdelta_C


	for t,word_value in enumerate(train_words):
		C[word_value] = input_sequence[(t-1)*M:(t*M)]

	print 'done on one full batch'


if __name__ == '__main__':
	categories = brown.categories() #List of all the catgories in the brown corpus
	words = brown.words(categories = 'adventure') #words take from a particular category of the brown corpus
	vocabulary = list(set(words))						#unique words in the corpus
	train_length,validation_length = int(0.6 * len(words)),int(0.2 * len(words)) 
	training_words = words[0:train_length] #60% of the corpus 
	validation_words = words[train_length : train_length + validation_length] #futher 20% of the corpus
	test_words = words[train_length + validation_length : len(words)] #remaining corpus
	training(training_words)