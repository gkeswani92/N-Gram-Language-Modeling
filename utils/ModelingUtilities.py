'''
Created on Sep 13, 2015

@author: gaurav
'''

import json
import os
import cPickle
import codecs
from nltk import word_tokenize

#Variable definitions relative to the path of the source files
base_path     = os.path.dirname(__file__)
genres        = ['children', 'crime', 'history']
training_file_counts = {'children': 6, 'crime': 6, 'history': 3}
training_path = base_path + '/books/train_books/'
test_path     = base_path + '/books/test_books/'

def getTokensForFile(path):
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

def serializeModelToDisk(model, ngram):
    '''
        Serialises the model of the ngram to its respective folder
    '''
    model_path = base_path + '/{0}/'.format(ngram)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    for genre, model in model.items():
        print("Serialising {0} model to disk".format(genre))
        f = open(model_path+genre,'w')
        
        #if ngram == 'Unigram':
        if 'Unigram' in ngram:
            json.dump(model, f)
        else:
            cPickle.dump(model, f)
        f.close()

def loadUnigramModels(dir_name='Unigram'):
    '''
        Loads the unigram models for all the genres from the JSON dump
    '''
    model_path = base_path + '/' + dir_name + '/'
    unigram_model = {}
    
    for genre in genres:
        f = open(model_path+genre,'r')
        unigram_model[genre] = json.load(f)
    
    return unigram_model

def loadBigramModels( dir_name='Bigram'):
    '''
        Loads the bigram models for all genres by unpickling 
    '''
    model_path = base_path + '/{0}/'.format(dir_name)
    bigram_model = {}
    
    for genre in genres:
        f = open(model_path+genre,'r')
        bigram_model[genre] = cPickle.load(f)
    
    return bigram_model
    

    
    
            
    
        