'''
Created on Sep 13, 2015

@author: gaurav
'''
from ModelingUtilities import genres
from scipy.stats       import rv_discrete
import pprint

def generateRandomSentenceFromUnigram(unigram_model, n=100):
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
        sentence = smartJoin([numberToWordMapping[genre][num] for num in genre_sample])
        
        randomUnigramSentences[genre] = str(sentence)
        
    print("Unigram random sentences:")
    pprint.pprint(randomUnigramSentences)
    return randomUnigramSentences

def generateRandomSentenceFromBigram(bigram_model, n=100, seed=None):
    '''
        Generating random sentences from the bigram model
    '''

    if not seed:
        seed = {'children':'<START>','crime':'<START>','history':'<START>'}
        #seed = {'children':'Suddenly','crime':'Inch','history':'Religion'}

    randomBigramSentences = {}
    for genre in genres:
        current_word = seed[genre]

        genre_sentence = [current_word]
        for word_ind in range(n):

            next_word_dict = bigram_model[genre][current_word]

            wordLevelNumberMapping = {}
        
            for i, key in enumerate(next_word_dict.keys()):
                wordLevelNumberMapping[i] = key

            next_word_num = rv_discrete(values=(wordLevelNumberMapping.keys(), next_word_dict.values())).rvs(size=1)
            current_word = wordLevelNumberMapping[next_word_num[0]]
            genre_sentence.append(current_word)

        randomBigramSentences[genre] = smartJoin(genre_sentence[1:]) # remove <START> before passing

    print("Bigram random sentences:")
    pprint.pprint(randomBigramSentences)
    return randomBigramSentences

def smartJoin(word_list):
    '''
        Because punctuation and other special characters are mixed in with the words,
        we can't just join everything using spaces. This makes the sentences more readable
        by eliminating spaces where they aren't needed.
    '''

    end_punctuation = set(['.','!','?'])
    middle_punctuation = set([',',';',':'])
    open_brace = set(['(','['])
    close_brace = set([')',']'])
    
    sentence_string = word_list[0]

    for i in range(1,len(word_list)):
        no_space = False
        current_word = word_list[i]
        prev_word = word_list[i-1]

        if "'" in current_word[0] and prev_word.isalpha(): # Contractions
            no_space = True
        elif current_word in middle_punctuation: # Comma/Colon/Semicolon
            no_space = True
        elif current_word in end_punctuation: # Sentence ending
            no_space = True
        elif current_word == "''" and prev_word in end_punctuation: # Quote ending
            no_space = True
        elif prev_word == "``" and current_word.isalpha(): # Quote starting
            no_space = True
        elif prev_word in open_brace: # Open brace
            no_space = True
        elif current_word in close_brace: # Close brace
            no_space = True

        if no_space:
            sentence_string += current_word
        else:
            sentence_string += ' '+current_word

    return sentence_string



    