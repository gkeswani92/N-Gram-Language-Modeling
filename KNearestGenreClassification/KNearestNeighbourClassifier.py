'''
Created on Sep 18, 2015

@author: gaurav
'''

from KNearestGenreClassification.ClassificationUtilities        import tfIdfTransform
from collections 										        import OrderedDict, Counter
from nltk.tokenize                                              import word_tokenize
import codecs
import numpy
import operator
import math

def KNearestNeighbourController(training_paths, training_labels, test_paths = None, test_labels = None, k = 1, tf_idf = False):
    '''
        Controller function to co-ordinate the K-Nearest neighbour algorithm
    '''
    #Returns the text per file and their labels once and for all
    training_text_dict = readTextFromFilePaths(training_paths)
    
    #If this is a validation run, we need to keep one book aside at a time and send
    #the rest of the them to create the training vectors
    if not test_paths:
        
        #Creating the vectors for the training set. This includes all but one book
        #which will be used for validation
        for validation_file in training_text_dict:
            print("Validation being done on: {0}".format(validation_file))
            
            #Creating the training vectors and validation vectors from the training dictionary
            vectors, _ = createTrainingVectors(training_text_dict)
            
            #Performing tf-idf transform on all the vectors
            tfidf_transform_dict = tfIdfTransform(vectors) if tf_idf else vectors
            
            #Separating the validation vector from the training one
            validation_vector = tfidf_transform_dict.pop(validation_file)
            training_vectors = tfidf_transform_dict
            
            #Finding the nearest neighbours by passing in a manual k
            nearest_neighbours = findNearestNeighbour(training_vectors, validation_vector, k)

            #Finding the genre with the concept of voting by neighbours           
            genre = detectClassOfTestSet(nearest_neighbours, training_labels, k)
            print("Known label: {0} Calculated Label: {1}".format(training_labels.get(validation_file,"Unknown"), genre))
            
    #Else clause gets activated in case the run is a final test run
    else:
        #Creating the training vectors from the training dictionary
        vectors, zero_vector = createTrainingVectors(training_text_dict)
    
        #Finding the genre of a test file one file at a time
        for test_file_path in test_paths:
            test_file = test_file_path.split('/')[-1]
            print("\nTest file is : {0}".format(test_file))

            #Creating the set of tokens for the test file
            test_text_dict = readTextFromFilePaths([test_file_path])
            
            #Running k nearest neighbours algorithms
            genre = runKNearestNeighbours(vectors, test_text_dict, test_file, training_labels, zero_vector, k, tf_idf)             
            print("Known label: {0} Calculated Label: {1}".format(test_labels.get(test_file,"Unknown"), genre))
            
            if tf_idf:
                vectors.pop(test_file)


def runKNearestNeighbours(training_vectors, tokenized_test_file, test_file, training_labels, zero_vector, k, tf_idf = False):
    
    #Creating the test vector with only those tokens that were a part of the training set
    training_vectors[test_file] = createTestVectors(tokenized_test_file[test_file], zero_vector)
    
    #Performing tf-idf transform on all the vectors
    tfidf_transform_dict = tfIdfTransform(training_vectors) if tf_idf else training_vectors
    
    #Seperating the test vector from the training vectors
    test_vector = tfidf_transform_dict.pop(test_file)
    training_vectors = tfidf_transform_dict
    
    #Finding the nearest neighbours by passing in a manual k
    nearest_neighbours = findNearestNeighbour(training_vectors, test_vector, k)
    
    #Running k nearest neighbours algorithms
    genre = detectClassOfTestSet(nearest_neighbours, training_labels, k)
    
    return genre  
            
def readTextFromFilePaths(paths):
    '''
        Reads in all text files and converts them to count vectors, represented by orderedDicts
    '''
    #Dictionary to map the file names to the text in each file
    tokenized_texts_dict = {}
    
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
    print("Creating vectors for training data")
    
    unique_words = []
    for filename, text in tokenized_texts_dict.iteritems():
        #print("Reading {0} and adding to unique word list".format(filename))
        unique_words.extend(word_tokenize(text))
    
    unique_words = set(unique_words)
    
    #Creating the initial vector with counts 0 for all training sets
    zero_vector = OrderedDict(zip(unique_words,[0]*len(unique_words)))
    print("Creating the zero vector")
    
    #For each training file, create an OrderedDict containing its word counts (together with zero counts),
    #and store it in a dict, indexed by its corresponding filename
    vectors = {}
    for filename, token_list in tokenized_texts_dict.iteritems():
        current_vector = zero_vector.copy()
        current_vector.update(Counter(word_tokenize(token_list)))
        vectors[filename] = current_vector
    
    return vectors, zero_vector

def createTestVectors(testset, zero_vector):
    '''
        Creating the test set depending on the frequency of words occuring in the 
        test set that are present in the zero_vector
    '''
    test_set_word_frequency = Counter(word_tokenize(testset))
    test_vector = OrderedDict((token, test_set_word_frequency.get(token,0)) for token in zero_vector)
    return test_vector

def computeDistance(vector1, vector2, method = 'cosine'):
    '''
        Computes the distance between two vector depending on the method that
        is passed in. Default method is cosine distance
    '''
    #Using cosine distance as a measure of distance
    if method == 'cosine':
        theta = numpy.dot(vector1, vector2) / numpy.sqrt(numpy.dot(vector1, vector1) * numpy.dot(vector2, vector2))  
        distance = 1 - theta
    
    #Using the euclidean distance measure
    elif method == 'euclidean':
        distance = euclideanDistance(vector1, vector2)
        
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
        distance_from_training_vectors[filename] = computeDistance(vector.values(), test_vector.values(), 'cosine')
    
    #Sorting the neighbours in ascending order based on their distance from the test vector
    test_neighbours = sorted(distance_from_training_vectors.iteritems(), key=operator.itemgetter(1))
    
    #Returning the k nearest neighbours of the test vector
    return test_neighbours[:k]

def detectClassOfTestSet(nearest_neighbours, training_labels, k):
    '''
        Takes in the nearest neighbours of a vector and its training labels
        to determine the class of the test set
    '''
    nearest_neighbour = []
    
    #Continue this loop until we find a genre that is maximum
    while True:
        genres_neighbours = [training_labels[book] for book, _ in nearest_neighbours]
        count_genres = Counter(genres_neighbours)
        nearest_neighbour = [genre for genre, count in count_genres.iteritems() if count == max(count_genres.values())]
        
        #If there is a majority, return the winning genre
        if len(nearest_neighbour) == 1:
            return nearest_neighbour[0]
        
        #If there is a tie, remove the last element from the nearest neighbours and vote again
        else:
            k = k - 1
            nearest_neighbours = nearest_neighbours[:k]

def euclideanDistance(instance1, instance2):
    '''
        Finds the euclidean distance between two vectors
    '''
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)  