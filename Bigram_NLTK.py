'''
Created on Sep 11, 2015

@author: gaurav
'''

import pprint
import nltk
import os

#Variable definitions relative to the path of the source files
base_path     = os.path.dirname(__file__)
genres        = ['children', 'crime', 'history']
training_path = base_path + '/books/train_books/'
test_path     = base_path + '/books/test_books/'

def generateBigramModels():
    '''
        Controller for the generation of the unigram models. 
        Iterates over the given genre folders and retrieves the unigram model
        to create the final unigram model dictionary
    '''
    
    #Get the bigrams in the corpus by the genre
    bigrams = {}
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        bigrams[genre] = getBigramsForGenre(training_path+genre)
    
    #Print this at your own risk :)
    #pprint.pprint(bigrams)
    
    #NLTK package conditional probabilities. Storing it like this would be ideal
    for key,value in nltk.ConditionalFreqDist(bigrams['children']).iteritems():
        print ((key,value))

def getBigramsForGenre(dir_path):
    '''
        Reads through the contents of a complete directory path and finds
        the bigrams present in a genre level corpus
    '''
    content = []
    genre_bigram = []
    
    for path in os.listdir(dir_path):
        
        #Reading the file's contents
        file_path = dir_path + '/' + path
        f = open(file_path,'r')
        print("Reading file at {0}".format(file_path))
        
        #Creating a list of all the tokens in the file
        for line in f.readlines():
            line = [x.strip() for x in line.split(' ') if x]
            content.extend(line)
            
        #All books in the same genre are aggregated into one bigram list
        genre_bigram.extend(list(nltk.bigrams(content)))
        break 
    
    return genre_bigram
    
generateBigramModels()
                
   
