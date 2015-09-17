'''
Created on Sep 13, 2015

@author: gaurav
'''
from ModelingUtilities import genres
from scipy.stats       import rv_discrete
import pprint

end_punctuation     = set(['.','!','?'])
middle_punctuation  = set([',',';',':'])
open_brace          = set(['(','['])
close_brace         = set([')',']'])
    
def generateRandomSentenceFromUnigram(unigram_model, n = None):
    '''
        Generating random sentences from the unigram model
    '''
    randomUnigramSentences = {} 
    
    #Creating a mapping of an integer to a word for random variable sampling at the genre level
    numberToWordMapping = {}
    for genre in genres:
        numberToWordMapping[genre] = dict((i, key) for i, key in enumerate(unigram_model[genre].keys()))
    
    #Creating a list of all the probabilities at a genre level
    numberToProbalityMapping =  dict((genre, unigram_model[genre].values()) for genre in genres)
    
    for genre in genres:
        current_token    = '<START>'
        current_sample   = []
        unigram_sentence = []
        count            = 0
         
        while (n and count<=n) or current_token not in end_punctuation:
            
            #Sampling on the probability distribution and creating a sample of size 1
            current_sample = rv_discrete( values=(numberToWordMapping[genre].keys(), numberToProbalityMapping[genre]) ).rvs(size=1)[0]
            
            #Converts the sampled number to its corresponding word
            current_token  = numberToWordMapping[genre][current_sample]
            
            #Contains all the tokens of the senetnce
            unigram_sentence.append(current_token)
            count += 1
            
        #Creating a sentence from the sample by matching the numbers to the words
        randomUnigramSentences[genre] = str(smartJoin(unigram_sentence))
        
    print("\nUnigram random sentences:")
    pprint.pprint(randomUnigramSentences)
    return randomUnigramSentences

def generateRandomSentenceFromBigram(bigram_model, seed=None, n = None):
    '''
        Generating random sentences from the bigram model
    '''
    randomBigramSentences = {}
    
    if not seed:
        seed = {'children':'<START>','crime':'<START>','history':'<START>'}

    for genre in genres:
        count = 0
        
        #If the seed is a single word, take it as it is. If it is a sentence, split it 
        #and take the last word
        current_token = seed[genre] if ' ' not in seed[genre] else seed[genre].split()[-1]

        #Even though the word being used as a seed is one word, the sentence should contain
        #the complete string from the beginning
        genre_sentence = seed[genre].split()
        
        while ( n and count<=n ) or current_token not in end_punctuation:
            
            successor_dict = bigram_model[genre][current_token]
            
            #Create a integer to word mapping of the possible next words
            numberToWordMapping = dict((i,key) for i, key in enumerate(successor_dict.keys()) )
            
            #Sampling on the probability distribution and creating a sample of size 1
            current_sample = rv_discrete(values=(numberToWordMapping.keys(), successor_dict.values())).rvs(size=1)[0]
            
            #Converts the sampled number to its corresponding word
            current_token = numberToWordMapping[current_sample]
            
            genre_sentence.append(current_token)
            count += 1
            
        #Smart joins the genre level sentence and stores it in the dictionary
        randomBigramSentences[genre] = smartJoin(genre_sentence) # remove <START> before passing

    print("\nBigram random sentences:")
    pprint.pprint(randomBigramSentences)
    return randomBigramSentences

def smartJoin(word_list):
    '''
        Because punctuation and other special characters are mixed in with the words,
        we can't just join everything using spaces. This makes the sentences more readable
        by eliminating spaces where they aren't needed.
    '''
    
    sentence_start_index = 1 if word_list[0] == '<START>' else 0
    sentence_string      = word_list[sentence_start_index]
    
    for i in range(sentence_start_index+1,len(word_list)):
        
        use_space    = True
        current_word = word_list[i]
        prev_word    = word_list[i-1] 

        if "'" in current_word[:2] and prev_word.isalpha(): #Contractions- do and n't would have a space in between by default. Removing it.
            use_space = False
            
        elif current_word in middle_punctuation: #Comma/Colon/Semicolon- there should be no space between a word and a comma
            use_space = False
            
        elif current_word in end_punctuation: #Sentence ending
            use_space = False
            
        elif current_word == "''" and prev_word in end_punctuation or prev_word in middle_punctuation: # Quote ending
            use_space = False
            
        elif prev_word == "``" and current_word.isalpha(): # Quote starting
            use_space = False
            
        elif prev_word in open_brace: # Open brace
            use_space = False
            
        elif current_word in close_brace: # Close brace
            use_space = False

        sentence_string += current_word if not use_space else ' ' + current_word

    return sentence_string



    