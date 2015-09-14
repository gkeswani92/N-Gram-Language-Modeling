'''
Created on Sep 13, 2015

@author: gaurav
'''
from ModelingUtilities import genres
from scipy.stats       import rv_discrete
import pprint
import pdb

def generateRandomSentenceFromUnigram( unigram_model, n = 30 ):
    '''
        Generating random sentences from the unigram model
    '''
    
    #Creating a mapping of an integer to a word for random variable sampling at the genre level
    numberToWordMapping = {}
    for genre in genres:
        genreLevelNumberMapping = {}
        
        for i, key in enumerate(unigram_model[genre].keys()):
            genreLevelNumberMapping[i] = key
        
        numberToWordMapping[genre] = genreLevelNumberMapping
    
    #Creating a list of all the probabilities at a genre level
    numberToProbalityMapping = {}
    for genre in genres:
        numberToProbalityMapping[genre] = unigram_model[genre].values()
    
    randomUnigramSentences = {}
    
    for genre in genres:
        
        #Sampling on the probability distribution 
        genre_sample = rv_discrete( values=(numberToWordMapping[genre].keys(), numberToProbalityMapping[genre]) ).rvs(size=n)
        
        #Creating a sentence from the sample by matching the numbers to the words
        sentence = ' '.join([numberToWordMapping[genre][num] for num in genre_sample])
        
        randomUnigramSentences[genre] = str(sentence)
        
    print("Unigram random sentences:")
    pprint.pprint(randomUnigramSentences)
    return randomUnigramSentences

def generateRandomSentenceFromBigram(bigram_model, n=60, seed=None):
    '''
        Generating random sentences from the bigram model
    '''

    if not seed:
        seed = {'children':'<START>','crime':'<START>','history':'<START>'}
        #seed = {'children':'Suddenly','crime':'Inch','history':'Religion'}

    genre_sentences = []
    for genre in genres:
        current_word = seed[genre]

        genre_sentence = [current_word]
        for word_ind in range(n-1):

            next_word_dict = bigram_model[genre][current_word]

            wordLevelNumberMapping = {}
        
            for i, key in enumerate(next_word_dict.keys()):
                wordLevelNumberMapping[i] = key

            next_word_num = rv_discrete(values=(wordLevelNumberMapping.keys(), next_word_dict.values())).rvs(size=1)
            current_word = wordLevelNumberMapping[next_word_num[0]]
            genre_sentence.append(current_word)

        print(' '.join(genre_sentence))



    