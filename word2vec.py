from nltk.corpus import brown
import numpy as np
import re

def init_vectors(dim):
	
	C = 0.01 * np.random.randn(V*dim)/np.sqrt(V*dim)
	return np.matrix(C)

def one_hot(word):

	ret = np.zeros(V,dtype = np.uint8)
	index = vocabulary.index(word)
	ret[index] = 1
	return ret,index

def negative_distribution(n):
	#Sample negative words from the vocabulary
	#Probability is defined by count(word)/|Vocabulary|
	distribution = np.power(np.array(freq),3.0/4)
	sample = np.random.choice(distribution)
	index = np.where(distribution == sample)[0]
	return index

def sigmoid(z,func = "logistic"):
	
	if func == "logistic":
		return 1/(1+np.exp(-z))
	elif func == "tanh":
		return (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))
		
def one_training_example(x1,x2,U,V):

	k = 20
	vc,vw = x1[0]*U,x2[0]*V
	first_term = sigmoid(vc*vw)
	derivatives = []
	for i in range(k):
		index_of_negative_sample = negative_distribution()
		vn = V[index_of_negative_sample]
		value_neg = sigmoid(vc*vn)
		derivatives.append(value_neg*vc)
		V[index_of_negative_sample] = V[index_of_negative_sample] - leaning_rate * (value_neg * vc)

	derivatives.append((first_term - 1)*vc)
	#This the objective function that needs to be minimised
	# E = -log(first_term) - log(second_term)
	
	error_derivatives_positive = (first_term - 1)*vc
	error_derivatives_negative = sum(derivatives)

	V[x2[0]] = V[x2[0]] - leaning_rate * ((first_term - 1) * vc)
	U[x1[0]] = U[x1[0]] - leaning_rate * error_derivatives_negative

	return U,V

	



def training(words,window = 2,hidden_units = 300,epoch = 1):
	
	""" 
	This method takes in the words and returns a learned distribution of the feature vectors of the words and
		the probability function over words.
	
	Paramters 

	Input 
	Parameter1 : list
			Words in the corpus

	Parameter2 : int
			Size of the window goes from -window_size to +window_size(default -2 to +2 leaving the centre word)
	
	Parameter3 : int
			The dimensions for each word vector denoted as the number of hidden units.
	
	Parameter4 : int
			It denotes the number of iterations through the training data

	Returns : Numpy Matrix
				Learned vector representation of the words in the corpus
	
	The objective function is to maximise the probabitlity of the words in the co-occuring the window
	"""

	U = init_vectors(hidden_units)
	V = init_vectors(hidden_units)
	V = V.T

	#Regular expression for all the Punctuations
	REGEX = "[a-zA-Z0-9]"

	iters = 0
	while iters < epoch:
		for i,word in enumerate(words):
			for j in range(-window,window+1):
				centre_word = words[i]
				try:
					outside_word = words[i+j]
				except IndexError:
					continue

				#if the outside_word or the centre_word is a punctuation break the loop and go to next centre_word
				# if there is a punctuation:
				# 	break

				#Maximise the probability for the centre_word and outside_word
				x1,x2 = one_hot(centre_word),one_hot(outside_word)
				U,V = one_training_example(x1,x2,U,V)
		iters+=1




if __name__ == '__main__':
	categories = brown.categories() #List of all the catgories in the brown corpus
	words = brown.words(categories = 'adventure') #words take from a particular category of the brown corpus
	vocabulary = list(set(words))
	V = len(vocabulary)						#unique words in the corpus
	freq = []
	for word in vocabulary:
		freq.append(words.count(word)/len(vocabulary))
	# train_length,validation_length = int(0.6 * len(words)),int(0.2 * len(words)) 
	# training_words = words[0:train_length] #60% of the corpus 
	# validation_words = words[train_length : train_length + validation_length] #futher 20% of the corpus
	# test_words = words[train_length + validation_length : len(words)] #remaining corpus
	training(words)