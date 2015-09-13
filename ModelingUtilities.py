'''
Created on Sep 13, 2015

@author: gaurav
'''

import json
import os
import cPickle

#Variable definitions relative to the path of the source files
base_path     = os.path.dirname(__file__)
genres        = ['children', 'crime', 'history']
training_path = base_path + '/books/train_books/'
test_path     = base_path + '/books/test_books/'
special_characters = '[~!@#$%^?&*()_,.+{}":;/\']+$123456789'

def serializeUnigramModelToDisk(model, ngram):
    '''
        Serialises the model of the ngram to its respective folder
    '''
    model_path = base_path + '/{0}/'.format(ngram)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    for genre, model in model.items():
        print("Serialising {0} model to disk".format(genre))
        f = open(model_path+genre,'w')
        
        if ngram == 'Unigram': 
            json.dump(model, f)
        else:
            cPickle.dump(model, f)
        f.close()
    
        