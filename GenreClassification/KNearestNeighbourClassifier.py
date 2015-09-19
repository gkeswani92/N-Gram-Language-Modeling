'''
Created on Sep 18, 2015

@author: gaurav
'''

from ModelCreation_SentenceGenerator.ModelingUtilities 	import genres, training_path, base_path
from collections 										import OrderedDict, Counter
from nltk.tokenize                                      import word_tokenize

def readTextAndVectorize(training_paths, test_paths):
	'''
	Reads in all text files and converts them to count vectors, represented by orderedDicts
	'''

	# Dictionary mapping each training file name to the list of word tokens which occur in that file
	tokenized_texts_dict = dict()
	for path in training_paths:
		f = codecs.open(file_path,'r','utf8', errors='ignore')
    	tokenized_texts_dict.update({path.split('/').[-1]: f.read()})
    	f.close()

    # Set of all unique words which occur in the training documents
    unique_words = set(word for word in token_list for token_list in tokenized_texts_dict.values())

    # Create an OrderedDict with the unique words as keys, and the values all being zero
    zero_vector = OrderedDict(zip(list(unique_words),[0 for _ in range(len(unique_words))]))
       
    # For each training file, create an OrderedDict containing its word counts (together with zero counts),
    # and store it in a dict, indexed by its corresponding filename
    training_dicts = dict()
    for filename,tokenlist in tokenized_texts_dict.iteritems():
    	this_dict = zero_vector.copy()
    	this_dict.update(Counter(tokenlist))
    	training_dicts.update({filename: this_dict})

    # Also create dict mapping filenames to labels

    # Create a separate dict mapping the test (or validation) filenames to OrderedDicts as well,
    # using the "zero_vector" constructed from the training files as a scaffold

