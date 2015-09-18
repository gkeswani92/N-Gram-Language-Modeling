'''
Created on Sep 10, 2015

@author: gaurav
'''

from GenerateUnigramModel    import generateUnigramModels
from GenerateBigramModel     import generateBigramModels
from ModelingUtilities       import loadUnigramModels, loadBigramModels
from RandomSentenceGenerator import generateRandomSentenceFromUnigram, generateRandomSentenceFromBigram
    
def main( create_model = False ):
    
    #Generate the unigram model for all the genres or load it from memory
    unigram_model = generateUnigramModels() if create_model else loadUnigramModels()
    
    #Generate random sentences from the unigram model which ends as soon as sentence end character is presented
    generateRandomSentenceFromUnigram(unigram_model)
    
    #Generate random sentences from the unigram model with sentence cap
    generateRandomSentenceFromUnigram(unigram_model, n=200)
    
    #Generate the bigram model for all the genres
    bigram_model = generateBigramModels() if create_model else loadBigramModels()
    
    #Generate random sentences from the bigram model with default seed and n=200
    generateRandomSentenceFromBigram(bigram_model, n=200)    
    
    #Generate random sentences from the bigram model with custom seed and n=100
    #Will consider <START> character as seed for history which has not been specified
    bigram_seed = {
                    'children':'sjbdsabdoisabdoisbdoias', 
                    'crime':'killed'
                  }
    generateRandomSentenceFromBigram(bigram_model, seed = bigram_seed, n=200)
    
if __name__ == '__main__':
    main()