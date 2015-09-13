'''
Created on Sep 13, 2015

@author: gaurav
'''

import re
import json
import os


#Variable definitions relative to the path of the source files
base_path     = os.path.dirname(__file__)
genres        = ['children', 'crime', 'history']
training_path = base_path + '/books/train_books/'
test_path     = base_path + '/books/test_books/'
special_characters = '[~!@#$%^?&*()_,.+{}":;/\']+$123456789'

def formatWordForQuality(word):
    '''
        Replaces the special characters in the passed word with None. This is
        done to weed out the unwanted characters
    '''
    
    word = str(word)
    
    #Removing the special characters from the words
    special_characters = '[~!@#$%^?&*()_,.+{}":;/\']+$123456789'
    #word = word.translate(None, special_characters)
    
    #Regex method to do the same
    word = re.sub(special_characters, '', word)
    
    #Converting everything to lower case
    word = word.lower()
    
    return word


def serializeUnigramModelToDisk(model, ngram):
    '''
        Serialises the model of the ngram to its respective folder
    '''
    model_path = base_path + '/{0}/'.format(ngram)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    for genre, model in model.items():
        print("Serialising {0} model to disk".format(genre))
        json.dump(model, open(model_path+genre,'w'))
    
        