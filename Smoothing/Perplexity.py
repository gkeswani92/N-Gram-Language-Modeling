'''
Created on Sep 21, 2015

@author: gaurav
'''
from utils.ModelingUtilities                              import test_path, genres, loadUnigramModels, getTokensForFile, loadBigramModels
from math                                                 import exp, log
from collections                                          import defaultdict
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
            book_tokens[path] = getTokensForFile(test_path + genre + '/' + path)
    
    #Computes the perplixity of every test corpus against each of the unigram models created
    book_perplexity = defaultdict(dict)
    for book, unigrams in book_tokens.iteritems():
        print('')
        for genre, model in unigram_model.iteritems():
            book_perplexity[book][genre] = computeUnigramPerplexity(model, unigrams)
            print("Perplexity of '{0}' book on {1} genre model: {2}".format(book,genre,book_perplexity[book][genre]))
    
    return book_perplexity
    
    
def getBigramPerplexity():
    '''
        Controller method to find the perplexity of all the test books with all the genres
    '''
    bigram_model = loadBigramModels()

    #Reads in the unigrams one file at a time and stores it with their bookname
    book_tokens = {}
    for genre in genres:
        print("\nReading test files for genre {0}".format(genre))
        for path in os.listdir(test_path + genre):
            book_tokens[path] = getTokensForFile(test_path + genre + '/' + path)

    # Construct Bigrams from Unigrams
    book_bigrams = {}
    for path, tokens in book_tokens.iteritems():
        book_bigrams[path] = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens)-1)]

    # Predict Bigram Perplexity
    book_perplexity = defaultdict(dict)
    for book, bigrams in book_bigrams.iteritems():
        for genre in genres:
            book_perplexity[book][genre] = computeBigramPerplexity(bigram_model[genre], bigrams)
            print("Perplexity of '{0}' book on {1} genre model: {2}".format(book, genre, book_perplexity[book][genre]))

    return book_perplexity


def computeUnigramPerplexity(genre_model, unigrams):
    '''
        Computes the perplexity of a given set of unigrams with the model that has been
        passed in
    '''
    unigram_probabilities = [ genre_model.get(unigram, genre_model['<UNKNOWN>']) for unigram in unigrams ]
    perplexity_value = exp(1.0/len(unigrams) * sum([ -log(1.0 * x) for x in unigram_probabilities]))
    return perplexity_value

def computeBigramPerplexity(bigram_model, bigrams):
    '''
        Computes the perplexity of a given set of unigrams with the model that has been
        passed in
    '''

    bigram_probabilities = []
    
    for bigram in bigrams:
        if bigram[0] in bigram_model:
            if bigram[1] in bigram_model:
                
                #If the bigram is one that was seen in the corpus
                if bigram[1] in bigram_model[bigram[0]].keys():
                    bigram_probabilities.append(bigram_model[bigram[0]][bigram[1]])
                    
                #If the first word was seen, but the second word wasn't seen with it
                else:
                    bigram_probabilities.append(bigram_model[bigram[0]]['<UNSEEN>'])
                  
            else:
                if '<UNKNOWN>' in bigram_model[bigram[0]]:
                    bigram_probabilities.append(bigram_model[bigram[0]]['<UNKNOWN>'])
                else:
                    bigram_probabilities.append(bigram_model[bigram[0]]['<UNSEEN>'])
                    
        elif bigram[1] in bigram_model:
            if bigram[1] in bigram_model['<UNKNOWN>']:
                bigram_probabilities.append(bigram_model['<UNKNOWN>'][bigram[1]])
            else:
                bigram_probabilities.append(bigram_model['<UNKNOWN>']['<UNSEEN>'])  
        else:
            bigram_probabilities.append(bigram_model['<UNKNOWN>']['<UNKNOWN>'])

    perplexity_value = exp(1.0/len(bigram_probabilities) * sum([ -log(1.0 * x) for x in bigram_probabilities]))
    
    return perplexity_value


if __name__ == '__main__':
    getUnigramPerplexity()
    getBigramPerplexity()