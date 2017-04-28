# mikolov_word2vec
This project is besed on the paper "Distributed Representations of Words and Phrases and their Compositionality" by Tomas Mikolov et al.

1. 	Create a virtual environment using anaconda(install anaconda2 if you do not have it installed)
		
		$conda create -n <env-name> python=2

2. 	Activate the virtual environment
		
		$source activate <env-name>

3. 	Install the required package
		
		$conda install --file requirements.txt

4. 	Install the corpus using nltk download
		
		$ipython
			>>>import nltk
			>>>nltk.download()

5.	Run the scipt word2vec.py to find the word representations
	
		$python word2vec.py

6.	The word representations are stored as dictionary where each key-value pair is a word(string) and its vector representation
	(numpy arrray) which is stored as a pickle file. 	
