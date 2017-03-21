from nltk.corpus import brown
import numpy as np

def training(words, learning_rate=0.5, M=30, hidden_units = 100,n = 3,**kwargs):
	
	""" This method takes in the words and returns a learned distribution of the feature vectors of the words.
	
	Parameters
	--------

	param1 : list of words
	
	param2 : float

		Learning rate is defined as the rate at which the model learns, higher the learning rate the model converges faster 
		or probably overshoots. 
	
	param3 : int 
		
		M is the length of vector for distributed representation of each word.
		
	param4 : int

		n = 3 corresponds to a trigram model.

	**kwargs

		Keyword arguments

	Returns
	-------

	C : numpy matrix

		Learned feature vector of words."""

	pass



if __name__ == '__main__':
	categories = brown.categories() #List of all the catgories in the brown corpus
	words = brown.words(categories = 'adventure') #words take from a particular category of the brown corpus
	train_length,validation_length = int(0.6 * len(words)),int(0.2 * len(words)) 
	training_words = words[0:train_length] #60% of the corpus 
	validation_words = words[train_length : train_length + validation_length] #futher 20% of the corpus
	test_words = words[train_length + validation_length : len(words)] #remaining corpus
	training(training_words)