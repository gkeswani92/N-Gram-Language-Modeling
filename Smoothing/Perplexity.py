'''
Created on Sep 21, 2015

@author: gaurav
'''

from ModelCreation_SentenceGenerator.ModelingUtilities import test_path, base_path, genres, training_path, serializeModelToDisk
from ModelCreation_SentenceGenerator.ModelingUtilities import loadUnigramModels
from math import exp, log
import os
import codecs
from nltk import word_tokenize

def getUnigramPerplexity():
    unigram_model = loadUnigramModels(smoothed=True)
    
    unigram_test_tokens = {}
    for genre in genres:
        unigram_test_tokens[genre] = getTestUnigrams(test_path+genre)
    
    for training_genre in genres:
        print("")
        for test_genre in genres:
            perplexity = computeUnigramPerplexity(unigram_model[training_genre],unigram_test_tokens[test_genre])
            print("\nPerplexity for {0} probability model on {1} data: {2}".format(training_genre,test_genre,str(perplexity)))
    

def getTestUnigrams(dir_path):
    '''
        Reads through the contents of a complete directory path and finds
        the frequency of each word to create a dictionary of word : count
    '''
    unigram_tokens = []
    for path in os.listdir(dir_path):
        file_path = dir_path + '/' + path
        print("Reading file at {0}".format(file_path))
        
        #Using nltk for tokenizing the word
        f = codecs.open(file_path,'r','utf8', errors='ignore')
        word_tokens = word_tokenize(f.read());
        f.close()
        
        unigram_tokens.extend(word_tokens)
   
    return unigram_tokens
    

def computeUnigramPerplexity(genre_model, test_unigrams):
    test_unigram_probabilities = [genre_model[unigram] for unigram in test_unigrams]
    perplexity_value = exp(1.0/len(test_unigrams)*sum([-log(1.0*x) for x in test_unigram_probabilities]))
    return perplexity_value
    

if __name__ == '__main__':
    getUnigramPerplexity()