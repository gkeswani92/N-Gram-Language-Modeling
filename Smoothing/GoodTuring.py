'''
Created on Sep 21, 2015

@author: gaurav
'''
from ModelCreation_SentenceGenerator.ModelingUtilities import test_path, base_path, genres, training_path, serializeModelToDisk
from ModelCreation_SentenceGenerator.GenerateUnigramModel import getUnigramFrequencyForGenre, getUnigramModelFeatures, createUnigramModel
from ModelCreation_SentenceGenerator.GenerateBigramModel import getBigramFrequencies 
from collections import Counter, defaultdict

def getUnigramFrequencies():
    '''
        Get dict of frequencies for all word types in all genres under the unigram model
        Make sure to add all of the zero-frequency word types extracted from the test files
    '''
    
    #Get the frequency of each word in the corpus
    unigram_frequencies = {}
    unigram_test_tokens = {}
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        unigram_frequencies[genre] = getUnigramFrequencyForGenre(training_path+genre)
        unigram_test_tokens[genre] = getUnigramFrequencyForGenre(test_path+genre).keys()
    
    unigram_test_tokens_flat = []
    for word_types in unigram_test_tokens.values():
        unigram_test_tokens_flat.extend(word_types)
    
    unigram_test_tokens_flat = set(unigram_test_tokens_flat)
    for genre,wordfreq_dict in unigram_frequencies.iteritems():
        missing_words = unigram_test_tokens_flat - set(wordfreq_dict.keys())
        wordfreq_dict.update(dict(zip(missing_words,[0]*len(missing_words))))
        unigram_frequencies[genre] = wordfreq_dict
        
    return unigram_frequencies
    
        
def applyGoodTuringSmoothingUnigram(n=5):
    '''
        Controller function to perform good-turing smoothing
    '''
    unigram_frequencies = getUnigramFrequencies()
    frequency_distribution = {}
    
    #Counter to determine how many unigrams occur 0 times, 1 times ...
    for genre, word_counts in unigram_frequencies.iteritems():
        frequency_distribution[genre] = Counter(word_counts.values())
    
    #Applying good turing smoothing on frequencies less than n
    smoothed_frequencies = defaultdict(dict)
    for genre, count_freq in frequency_distribution.iteritems():
        for i in range(0,n):
            smoothed_frequencies[genre][i] = (i + 1) * (float(count_freq[i+1])/count_freq[i])
    
    for genre, word_counts in unigram_frequencies.iteritems():
        for token, count in word_counts.iteritems():
            if count < n:
                word_counts[token] = smoothed_frequencies[genre][count]
    
    #Get the word type and token count for the smoothed frequencies
    unigram_features = getUnigramModelFeatures(unigram_frequencies)
    print("\n Unigram Features (Word Types, Work Tokens) {0} \n".format(unigram_features))
    
    #Creating the unigram model i.e. calculating the probabilities of the unigrams
    unigram_model = createUnigramModel(unigram_frequencies, unigram_features)
     
    #Storing the model on the disk in JSON format
    serializeModelToDisk(unigram_model, 'SmoothedUnigram')
    
    
if __name__ == '__main__':
    applyGoodTuringSmoothingUnigram()