'''
Created on Sep 9, 2015

@author: gaurav
'''

from collections import defaultdict
from itertools   import count
import pprint 
import os
import re
import codecs

#Variable definitions relative to the path of the source files
base_path     = os.path.dirname(__file__)
genres        = ['children', 'crime', 'history']
training_path = base_path + '/books/train_books/'
test_path     = base_path + '/books/test_books/'
special_characters = '[~!@#$%^?&*()_,.+{}":;/\']+$123456789'


def generateUnigramModels():
    '''
        Controller for the generation of the unigram models. 
        Iterates over the given genre folders and retrieves the unigram model
        to create the final unigram model dictionary
    '''
    
    #Get the frequency of each word in the corpus
    unigram_frequencies = {}
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        unigram_frequencies[genre] = getUnigramFrequencyForGenre(training_path+genre)
    
    #Get the word type and token count for the corpus
    unigram_features = getUnigramModelFeatures(unigram_frequencies)
    pprint.pprint("\nUnigram Features (Word Types, Work Tokens) {0}".format(unigram_features))
    
    unigram_model = createUnigramModel(unigram_frequencies, unigram_features)
    
    return unigram_model

def getUnigramFrequencyForGenre(dir_path):
    '''
        Reads through the contents of a complete directory path and finds
        the frequency of each word to create a dictionary of word : count
    '''
    word_frequency = defaultdict(int)
    
    for path in os.listdir(dir_path):
        file_path = dir_path + '/' + path
        print("Reading file at {0}".format(file_path))
        
        f = codecs.open(file_path,'r','utf-8')
        for line in f.readlines():
            for word in line.split():
                mod_word = word.strip().lower()
                word_frequency[removeSpecialCharacters(mod_word)] += 1
                
    return word_frequency    

def removeSpecialCharacters(word):
    '''
        Replaces the special characters in the passed word with None. This is
        done to weed out the unwanted characters
    '''
    
    #translate works like this only in Python 2.x
    return word.translate(None, special_characters)
    
    #return re.sub(special_characters, '', word)
        
def getUnigramModelFeatures(unigram_models):
    '''
        Creates a dictionary of genre : (word_types, word_tokens)
    '''
    unigram_features = {}
    for genre, model in unigram_models.items():
        word_types  = len(model.keys())
        word_tokens = 0
        
        #Adding the count of each word to the token count
        for count in model.values():
            word_tokens += count
        
        unigram_features[genre] = (word_types, word_tokens)
        
    return unigram_features

def createUnigramModel(unigram_frequencies, unigram_features):
    '''
        Uses the word frequency and the work token count to create the 
        unigram model per corpus
    '''
    unigram_model = {}
    
    for genre, frequencies in unigram_frequencies.items():
        token_count = unigram_features[genre][1]
        unigram_model[genre] = {}
        
        for word, frequency in frequencies.items():
            unigram_model[genre][word] = frequency * 1.0 / token_count
            
    return unigram_model
        
    
            
        