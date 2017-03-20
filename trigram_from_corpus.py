import nltk
from nltk.util import ngrams
from nltk.corpus import brown #Load the brown corpus

#Run download for the first time to download the nltk stuff
#nltk.download()

ngram = 3 #A trigram model for trigram training data
#sentences = brown.sents(categories = 'adventure')
words = brown.words(categories = 'adventure') #Get the words from the test corpus

# print len(sentences),len(words)

#Store the model to a text file

"""Punctuations in a corpus helps to understand the structuring of sentences hence must be used in word tokenizations.
But cannot be used in a trigram data model hence they are to be removed"""

punctuations = ['.',',','?','\'','\"',"!",":",";",'/']

text_file = open('trigram_adventure.txt','w')
for i,word in enumerate(words):
	if i > len(words)-ngram:
		break

	#continue if the word is punctuation	 
	if word in punctuations or words[i+1] in punctuations or words[i+2] in punctuations:
		continue
	value = words[i] + '\t' + words[i+1] + '\t' + words[i+2] + '\n'
	text_file.write(value)

text_file.close()