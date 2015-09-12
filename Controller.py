'''
Created on Sep 10, 2015

@author: gaurav
'''
import pprint
from GenerateUnigramModel import generateUnigramModels, getUnigramModelFeatures

special_characters = '[~!@#$%^?&*()_,.+{}":;/\']+$123456789'

def removeSpecialCharacters(word):
    '''
        Replaces the special characters in the passed word with None. This is
        done to weed out the unwanted characters
    '''
    
    #translate works like this only in Python 2.x
    return word.translate(None, special_characters)
    
    #return re.sub(special_characters, '', word)

def main():
    
    #Generate the unigram model for all the genres
    unigram_model = generateUnigramModels()
    #pprint.pprint(unigram_model)
  
if __name__ == '__main__':
    main()