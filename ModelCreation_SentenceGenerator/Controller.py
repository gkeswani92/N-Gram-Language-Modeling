'''
Created on Sep 10, 2015

@author: gaurav
'''

from ModelCreation_SentenceGenerator.GenerateUnigramModel    import generateUnigramModels
from ModelCreation_SentenceGenerator.GenerateBigramModel     import generateBigramModels
from utils.ModelingUtilities                                 import loadUnigramModels, loadBigramModels
from ModelCreation_SentenceGenerator.RandomSentenceGenerator import generateRandomSentenceFromUnigram, generateRandomSentenceFromBigram
    
def main():
    
    #Generate the unigram model for all the genres
    _ = generateUnigramModels()
    
    #Generate the bigram model for all the genres
    #_ = generateBigramModels( random_sentence  = False )
    
if __name__ == '__main__':
    main()