'''
Created on Sep 18, 2015

@author: gaurav
'''

import math
import numpy
import operator

def tfIdfTransform(vector_dict):
    '''
        Performing TF IDF (Token Frequency - Inverse Document Frequency) transformation
        on the data to reduce the weights of those words that appear frequently across
        all genres since they do not help any genre in standing out from the other
    '''
    transformed_vector_dict = {}
    
    for filename, vector in vector_dict.iteritems():
        
        #Normalising the frequencies because the size of the documents may differ
        #and so it is the relative frequency that matters instead of the absolute
        #frequency
        total_tokens      = float(sum(vector.values()))
        normalized_vector = dict((token, count/total_tokens) for token, count in vector.iteritems())
        
        #Performing the tf-idf transformation to give more weight to those words 
        #help a document to stand out from the others. This will reduce the importance
        #of the effect of words like "The" which will occur in all documents but
        #wont help us differentiate one document from another
        tfidf_transform_vector = {}
        for token, normalized_count in normalized_vector.iteritems():
            
            #Finding the number of documents that this token occurs in 
            no_of_documents_token_occurs  = sum([1 if possible_dict.get(token,0) else 0 for possible_dict in vector_dict.values()])

            #Formula for transformed frequency = Frequency in current vector x log(Total no of documents/Total number of documents the word occurs in)
            tfidf_transform_vector[token] =  normalized_count * math.log(len(vector_dict) / no_of_documents_token_occurs)
            
        transformed_vector_dict[filename] = tfidf_transform_vector
    
    return transformed_vector_dict

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
