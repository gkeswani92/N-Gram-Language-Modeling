'''
Created on Sep 18, 2015

@author: gaurav
'''

from collections 										import OrderedDict, Counter
from nltk.tokenize                                      import word_tokenize
from copy import deepcopy
import codecs
import numpy
import operator

def KNearestNeighbourController(training_paths, test_paths = None, validation = True, k = 1):
    '''
        Controlled function to co-ordinate the K-Nearest neighbour algorithm
    '''
    #Returns the text per file and their labels once and for all
    training_text_dict = readTextFromFilePaths(training_paths)
    
    #If this is a validation run, we need to keep one book aside at a time and send
    #the rest of the them to create the training vectors
    if validation:
        print("K-Nearest Neighbout validation run")
        
        #Creating the vectors for the training set. This includes all but one book
        #which will be used for validation
        for validation_file in training_text_dict:
            
            #Making a copy of the total text set, removing the validation file text
            #and passing the rest to create training vectors
            current_text_dict = deepcopy(training_text_dict)
            test_text         = current_text_dict.pop(validation_file)  #Contains the text of the current validation file
            training_dict     = current_text_dict                       #Contains file names of all files and their text except the validation file
            
            #Creating the training vectors from the training dict
            training_vectors, zero_vector = createTrainingVectors(training_dict)
            
            #Create the test vector depending on the words that were a part of training set
            test_vector = createTestVectors(test_text, zero_vector)
            
            #Finding the nearest neighbours by passing in a manual k
            nearest_neighbours = findNearestNeighbour(training_vectors, test_vector, k)
            
            print(nearest_neighbours)
            
            break

    
def readTextFromFilePaths(paths):
    '''
        Reads in all text files and converts them to count vectors, represented by orderedDicts
    '''
    #Dictionary to map the file names to the text in each file
    tokenized_texts_dict = {}
    print("Reading the text from the provided paths")
    
    for path in paths:
        
        #Storing the data against the file name
        f = codecs.open(path,'r','utf8',errors='ignore')
        tokenized_texts_dict[path.split('/')[-1]] = f.read()
        f.close()
        
    return tokenized_texts_dict

def createTrainingVectors(tokenized_texts_dict):
    '''
        Given the filenames and their contents, this methods creates the training 
        vectors by creating a unique list of all words together in the training
        set
    '''
    print("Creating vectors for the training data")
    
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
        
    return training_vectors, zero_vector

def createTestVectors(testset, zero_vector):
    '''
        Creating the test set depending on the frequency of words occuring in the 
        test set that are present in the zero_vector
    '''
    test_set_word_frequency = Counter(word_tokenize(testset))
    test_vector = dict((token, count) for token, count in test_set_word_frequency.iteritems() if token in zero_vector)
    return test_vector

def computeDistance(vector1, vector2, method = 'cosine'):
    '''
        Computes the distance between two vector depending on the method that
        is passed in. Default method is cosine distance
    '''

    if method == 'cosine':
        theta = numpy.dot(vector1, vector2)
        distance = 1 - theta
        
    return distance

def findNearestNeighbour(training_vectors, test_vector, k):
    '''
        Calculate distance of the test vector from all the training vectors and 
        return the k-nearest neighbours
        
        Expected format: 
        training_vectors = {'children.text': {'a': 0.0, 'c': 0.0, 'b': 0.0, 'd': 0.0}, 
                            'history.text': {'a': 0.0, 'c': 0.0, 'b': 0.0, 'd': 0.0}}
                            
        test_vector      = {'a': 0.0, 'c': 0.0, 'b': 0.0, 'd': 0.0}
        k                = 1 (any integer)
    '''
    
    #Computer distance of the test vector from all the training vectors
    distance_from_training_vectors = {}
    for filename, vector in training_vectors.iteritems():
        distance_from_training_vectors[filename] = computeDistance(vector.values(), test_vector.values())
    
    #Sorting the neighbours in ascending order based on their distance from the test vector
    test_neighbours = sorted(distance_from_training_vectors.iteritems(), key=operator.itemgetter(1))
    
    #Returning the k nearest neighbours of the test vector
    return test_neighbours[:k]

    