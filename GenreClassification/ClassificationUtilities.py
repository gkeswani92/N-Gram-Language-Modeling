'''
Created on Sep 18, 2015

@author: gaurav
'''

from ModelCreation_SentenceGenerator.ModelingUtilities import genres
from copy import deepcopy
import math
import os

def getFileListAndLabels(path):
    '''
        Gets the list of all files in the under the training/test path depending 
        on what is passed. 
        
        It also creates a dictionary of file name: genre
    ''' 
    file_paths = []
    genre_labels = {}
    
    for genre in genres:
        dir_path = path + genre
        for file_name in os.listdir(dir_path):
            file_paths.append(dir_path+'/'+file_name)
            genre_labels[file_name] = genre
            
    print(genre_labels)
    return file_paths, genre_labels
    
def tfIdfTransform(vector_dict):
    '''
        Performing TF IDF (Token Frequency - Inverse Document Frequency) transformation
        on the data to reduce the weights of those words that appear frequently across
        all genres since they do not help any genre in standing out from the other
    '''
    print("Performing TF-IDF transformation on the vectors")
    transformed_vector_dict = {}
    
    for filename, vector in vector_dict.iteritems():
        
        #Normalising the frequencies because the size of the documents may differ
        #and so it is the relative frequency that matters instead of the absolute
        #frequency
        total_tokens      = float(sum(vector.values()))
        normalized_vector = dict((token, float(count)/total_tokens) for token, count in vector.iteritems())
        
        #Performing the tf-idf transformation to give more weight to those words 
        #help a document to stand out from the others. This will reduce the importance
        #of the effect of words like "The" which will occur in all documents but
        #wont help us differentiate one document from another
        tfidf_transform_vector = {}
        for token, normalized_count in normalized_vector.iteritems():
            
            #Finding the number of documents that this token occurs in 
            no_of_documents_token_occurs  = sum([1 if possible_dict.get(token, False) else 0 for possible_dict in vector_dict.values()])
            
            #Formula for transformed frequency = Frequency in current vector x log(Total no of documents/Total number of documents the word occurs in)
            tfidf_transform_vector[token] =  normalized_count * math.log(float(len(vector_dict)) / no_of_documents_token_occurs)
            
        transformed_vector_dict[filename] = tfidf_transform_vector
    
    return transformed_vector_dict

def getTrainingAndTestSetText(training_text_dict, validation_file):
    '''
        Making a copy of the total text set, removing the validation file text
        and passing the rest to create training vectors
    '''
    current_text_dict = deepcopy(training_text_dict)
    test_text         = current_text_dict.pop(validation_file)  #Contains the text of the current validation file
    training_dict     = current_text_dict      
    
    return training_dict, test_text