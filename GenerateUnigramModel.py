'''
Created on Sep 9, 2015

@author: gaurav
'''

from ModelingUtilities import genres, training_path, serializeModelToDisk
from nltk.tokenize     import word_tokenize
from collections       import defaultdict, Counter
import os
import codecs

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
    print("\n Unigram Features (Word Types, Work Tokens) {0} \n".format(unigram_features))
    
    #Creating the unigram model i.e. calculating the probabilities of the unigrams
    unigram_model = createUnigramModel(unigram_frequencies, unigram_features)
     
    #Storing the model on the disk in JSON format
    serializeModelToDisk(unigram_model, 'Unigram')
    
    return unigram_model

def getUnigramFrequencyForGenre(dir_path, use_nltk = True):
    '''
        Reads through the contents of a complete directory path and finds
        the frequency of each word to create a dictionary of word : count
    '''
    word_frequency = defaultdict(int)
    
    for path in os.listdir(dir_path):
        file_path = dir_path + '/' + path
        print("Reading file at {0}".format(file_path))
        
        #Using nltk for tokenizing the word
        f = codecs.open(file_path,'r','utf8', errors='ignore')
        words = word_tokenize(f.read());
        f.close()
        
        #Creating a frequency chart of the word occurences
        word_frequency = Counter(words)
   
    return word_frequency    
        
def getUnigramModelFeatures(unigram_models):
    '''
        Creates a dictionary of genre : (word_types, word_tokens) where word
        types are the unique words while work tokens is the total count of the 
        words in the corpus.
    '''
    unigram_features = {}
    for genre, model in unigram_models.items():
        word_types  = len(model.keys())
        word_tokens = sum(model.values())
        unigram_features[genre] = (word_types, word_tokens)
        
    return unigram_features

def createUnigramModel(unigram_frequencies, unigram_features):
    '''
        Uses the word frequency and the work token count to create the 
        unigram model per genre i.e genre : { word1 : probability1, word2 : 
        probability2 }
    '''
    unigram_model = {}
    
    for genre, frequencies in unigram_frequencies.items():
        
        #Number of tokens for the particular genre is used to calculate probabilities of a word occuring in that genre
        token_count = unigram_features[genre][1]
        
        #Creating the model at the genre level
        unigram_model[genre] = dict((word, frequency * 1.0 / token_count) for word, frequency in frequencies.items())
        
    return unigram_model

            
        
