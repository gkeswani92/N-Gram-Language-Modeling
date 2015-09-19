'''
Created on Sep 18, 2015

@author: gaurav
'''

from ModelCreation_SentenceGenerator.ModelingUtilities 	import genres, training_path, base_path
from collections 										import OrderedDict, Counter
from nltk.tokenize                                      import word_tokenize
import codecs

def readTextAndLabels(training_paths, test_paths):
    '''
        Reads in all text files and converts them to count vectors, represented by orderedDicts
    '''
    #Variable to maintain the label for each of the training files that was read in
    book_genre_label = {}
    
    #Dictionary to map the file names to the text in each file
    tokenized_texts_dict = {}
    for path in training_paths:
        
        #Storing the data against the file name
        f = codecs.open(path,'r','utf8',errors='ignore')
        tokenized_texts_dict[path.split('/')[-1]] = f.read()
        print(path.split('/')[-1])
        f.close()
        
        #Storing the label of the file that was read in
        book_genre_label[path] = path.split('/')[-2]
        
    return tokenized_texts_dict, book_genre_label

def createTrainingVectors(tokenized_texts_dict):
    '''
        Given the filenames and their contents, this methods creates the training 
        vectors by creating a unique list of all words together in the training
        set
    '''
    #Creating a set of all the words in the training set
    unique_words = set([token for text in tokenized_texts_dict.values() for token in text])
    
    #Creating the initial vector with counts 0 for all training sets
    zero_vector = OrderedDict(zip(unique_words,[0]*len(unique_words)))
    
    #For each training file, create an OrderedDict containing its word counts (together with zero counts),
    #and store it in a dict, indexed by its corresponding filename
    training_vectors = {}
    for filename, token_list in tokenized_texts_dict.iteritems():
        current_vector = zero_vector.copy()
        current_vector.update(Counter(token_list))
        training_vectors[filename] = current_vector
        
    return training_vectors


    