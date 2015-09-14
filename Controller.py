'''
Created on Sep 10, 2015

@author: gaurav
'''

from GenerateUnigramModel import generateUnigramModels
from GenerateBigramModel  import generateBigramModels
from ModelingUtilities    import loadUnigramModels, loadBigramModels
from RandomSentenceGenerator import generateRandomSentenceFromUnigram
    
def main( create_model = False ):
    
    
        #Generate the unigram model for all the genres or load it from memory
        unigram_model = generateUnigramModels() if create_model else loadUnigramModels()
        
        #Generate random sentences from the unigram model
        generateRandomSentenceFromUnigram(unigram_model)
        
        #Generate the bigram model for all the genres
        bigram_model = generateBigramModels() if create_model else loadBigramModels()
    
if __name__ == '__main__':
    main()