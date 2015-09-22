'''
Created on Sep 9, 2015

@author: gaurav
'''

from utils.ModelingUtilities  import genres, training_path, serializeModelToDisk
from Smoothing.GoodTuring     import applyGoodTuringSmoothing
from nltk.tokenize            import word_tokenize
from collections              import Counter
import os
import codecs

def generateUnigramModels():
    '''
        ControllerModelAndRandomSentence for the generation of the unigram models. 
        Iterates over the given genre folders and retrieves the unigram model
        to create the final unigram model dictionary
    '''
    #Get the frequency of each word in the corpus
    unigram_frequencies = getUnigramFrequenciesforTrainingSet()
                
    #Get the word type and token count for the corpus
    unigram_features = getUnigramModelFeatures(unigram_frequencies)
    print("\n Unigram Features (Word Types, Work Tokens) {0} \n".format(unigram_features))
    
    #Returns the frequency distributions with all tokens with frequency 1 replacedby <UNKNOWN>
    unigram_features_unknown_words = handleUnknownWords(unigram_frequencies)
    unigram_frequencies = {} #Releasing the unused memory to speed up the program
    
    #Performing Good Turing Smoothing
    smoothed_frequencies = applyGoodTuringSmoothing(unigram_features_unknown_words, n = 5)
    
    #Creating the unigram model i.e. calculating the probabilities of the unigrams
    unigram_model = createUnigramModel(smoothed_frequencies, unigram_features)
     
    #Storing the model on the disk in JSON format
    serializeModelToDisk(unigram_model, 'Unigram')
    
    return unigram_model

def getUnigramFrequenciesforTrainingSet():
    '''
        Wrapper method to get the unigram frequency distribution across all 
        genres
    '''
    unigram_frequencies = {}
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        word_list = []
        
        #Reads in the unigrams one file at a time
        for path in os.listdir(training_path + genre):
            word_list.extend(getUnigramsForFile(training_path + genre + '/' + path))
        
        #Creating a counter of the frequencies at the genre level
        unigram_frequencies[genre] = Counter(word_list)
    
    return unigram_frequencies

def getUnigramsForFile(path):
    '''
        Reads through the contents of a file and returns the individual tokens
        as a list
    '''
    print("Reading file at {0}".format(path))
    
    #Using nltk for tokenizing the word
    f = codecs.open(path,'r','utf8', errors='ignore')
    words = word_tokenize(f.read());
    f.close()
    
    return words
        
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

def handleUnknownWords(unigram_features):
    '''
        Replaces all tokens with the frequency 1 as <UNKNOWN> and
        aggregates them into one entry at the genre level
    '''
    modified_unigram_frequency = {}
    for genre, frequency_dist in unigram_features.items():
        count = 0
        
        for token, frequency in frequency_dist.items():
            if frequency == 1:
                del frequency_dist[token]
                count += 1
        
        frequency_dist['<UNKNOWN>'] = count
        modified_unigram_frequency[genre] = frequency_dist
    
    return modified_unigram_frequency

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

            
        
