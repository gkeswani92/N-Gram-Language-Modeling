'''
Created on Sep 10, 2015

@author: gaurav
'''
import pprint
from GenerateUnigramModel import generateUnigramModels, getUnigramModelFeatures

def main():
    
    #Generate the unigram model for all the genres
    unigram_model = generateUnigramModels()
    #pprint.pprint(unigram_model)
  
   
    
if __name__ == '__main__':
    main()