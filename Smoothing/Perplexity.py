'''
Created on Sep 21, 2015

@author: gaurav
'''
from ModelCreation_SentenceGenerator.GenerateUnigramModel import getUnigramsForFile
from utils.ModelingUtilities                              import test_path, genres, loadUnigramModels
from math                                                 import exp, log
from collections                                          import defaultdict
import pprint
import os

def getUnigramPerplexity():
    '''
        Controller method to find the perplexity of all the test books with all the genres
    '''
    unigram_model = loadUnigramModels()
    
    #Reads in the unigrams one file at a time and stores it with their bookname
    book_tokens = {}
    for genre in genres:
        print("\nReading test files for genre {0}".format(genre))
        for path in os.listdir(test_path + genre):
            book_tokens[path] = getUnigramsForFile(test_path + genre + '/' + path)
    
    book_perplexity = defaultdict(dict)
    for book, unigrams in book_tokens.iteritems():
        for genre, model in unigram_model.iteritems():
            book_perplexity[book][genre] = computeUnigramPerplexity(model, unigrams)
    
    pprint.pprint(book_perplexity)
    
    
def computeUnigramPerplexity(genre_model, unigrams):
    '''
        Computes the perplexity of a given set of unigrams with the model that has been
        passed in
    '''
    unigram_probabilities = [ genre_model.get(unigram, genre_model['<UNKNOWN>']) for unigram in unigrams ]
    perplexity_value = exp(1.0/len(unigrams) * sum([ -log(1.0 * x) for x in unigram_probabilities]))
    return perplexity_value


if __name__ == '__main__':
    getUnigramPerplexity()