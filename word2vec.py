from nltk.corpus import brown
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def init_vectors(dim):
	
	"""Initialising the matrices using a random samples of a normal distribution
	
	args
	-----
	Parameter : int;the length of the dimensiions for each word vector
	
	Returns : Numpy Matrix; Initialised words vectors for each word in the vocabulary	
	"""
	C = 0.01 * np.random.randn(len_vocab,dim)/np.sqrt(len_vocab*dim)
	return np.matrix(C)

def one_hot(word):

	"""One hot representation of the word where the index for the word is a one and other indices are zeros
	
	args
	----
	Parameter : string, The word to match in the Vocabulary to be represented as a one-hot vector
	
	Returns : Numpy array,index; The one hot vector and the index of the word in the vocabulary
	"""
	  
	ret = np.zeros(len_vocab,dtype = np.uint8)
	index = vocabulary.index(word)
	ret[index] = 1
	return ret,index

def negative_distribution():
	"""
	Sample negative words from the vocabulary
	Probability is a function of count(word)/|Vocabulary| to power of 3/4
	
	Returns a sample according to the Probabibilty distribution 
	"""
	index = np.random.randint(low=0, high=table_size, size=1)

	return table[index][0]

def sigmoid(z,func = "logistic"):
	
	#Standard sigmoid functions
	if func == "logistic":
		return 1/(1+np.exp(-z))
	elif func == "tanh":
		return (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))
		
def one_training_example(x1,x2,U,V):

	#Gradient Descent on both the matrices for the input word and the outside words for a single training example
	#A recursive function to iteratively update the word matrices
	k = 20
	learning_rate = 0.01
	vc,vw = x1[0]*U,x2[0]*V

	first_term = sigmoid(np.multiply(vc,vw))
	derivatives = []
	for i in range(k):
		index_of_negative_sample = negative_distribution()

		if re.search(REGEX,vocabulary[index_of_negative_sample]):
			continue
		else:
			vn = V[index_of_negative_sample]
			value_neg = sigmoid(np.multiply(vc,vn))
			temp = np.multiply(value_neg,vc)
			derivatives.append(temp)
			V[index_of_negative_sample] = V[index_of_negative_sample] - learning_rate * temp

	derivatives.append(np.multiply((first_term - 1),vc))

	#This the objective function that needs to be minimised
	# E = -log(first_term) - log(second_term)
	
	error_derivatives_positive = np.multiply((first_term - 1),vc)
	error_derivatives_negative = sum(derivatives)

	V[x2[0]] = V[x2[0]] - learning_rate * error_derivatives_positive
	U[x1[0]] = U[x1[0]] - learning_rate * error_derivatives_negative

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
	
	The objective is to maximise the probabitlity of the words in the co-occuring the window
	"""

	U = init_vectors(hidden_units)
	V = init_vectors(hidden_units)

	iters = 0
	while iters < epoch:
		for i,word in enumerate(words):
			# print word
			#if the centre word is a punctuation go to the next word
			if re.search(REGEX,word):
				print 'Match!',word
				break
			
			else:
				for j in range(-window,window+1):
					centre_word = word
					try:
						outside_word = words[i+j]

					except IndexError:
						continue

					#if the outside_word or the centre_word is a punctuation break the loop and go to next outside word
					if re.search(REGEX,outside_word):
						print 'MATCH!',outside_word
						break

					#Maximise the probability for the centre_word and outside_word
					x1,x2 = one_hot(centre_word),one_hot(outside_word)
					U,V = one_training_example(x1,x2,U,V)
		
		iters+=1

	return U


if __name__ == '__main__':
	#Regular expression for all the Punctuations
	REGEX = "[^a-zA-Z0-9]"
	
	#List of all the catgories in the brown corpus and store all the words from the corus in a list
	categories = brown.categories()
	words = []
	for category in categories:
		words_temp = brown.words(categories = category) #words taken from a particular category of the brown corpus
		words.extend(words_temp)

	#Counter object to calculate the frequency of each word in the corpus
	count = Counter(words)
	vocabulary = count.keys() #List of all the words in the vocabulary
	len_vocab = len(vocabulary)
	
	#Frequency of each word in the corpus to form a distribution for the negative sampling 
	freq = np.float32(np.array(count.values()))
	numerator_distribution = np.power(freq,0.75) #Frequency raised to the power 3/4 which is empirical
	norm_term = sum(numerator_distribution)  	#Normalisisng factor
	distribution = numerator_distribution/norm_term
	
	#The way to form to select a word is from a table of word indices and store word indices multiple times according to the word
	#frequency given by formula freq(word)/len(vocabulary) times the table_size and sample a random integer from the (0,len(table)
	#and take the index at location of sample which is the index of the sampled word.
	#Refer the negative_sampling() method.
	table_size = int(1e8)
	table = np.zeros(table_size,dtype = np.int16)
	previous_j = 0
	for i,value in enumerate(distribution):
		j = int(table_size*value)
		table[previous_j:previous_j+j] = i
		previous_j = previous_j+j

	word_representations = training(words)
	
	#Word dictionary where key is the word and value is the word vector.
	word_dict = {}
	for i,word in enumerate(vocabulary):
		word_dict[word] = word_representations[i]

	#Save the dictionary as a pickle file
	joblib.dump(word_dict,'word.pkl',compress = 3)
