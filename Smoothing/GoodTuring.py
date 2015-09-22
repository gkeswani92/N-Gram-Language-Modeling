'''
Created on Sep 21, 2015

@author: gaurav
'''
from collections import Counter, defaultdict

def applyGoodTuringSmoothing(unigram_frequencies, n = 5):
    '''
        Takes the frequency distribution as an input and returns the smoothed
        frequency distribution
    '''
    #Counter to determine how many unigrams occur 0 times, 1 times, 2 times ...
    frequency_distribution= {}
    for genre, word_counts in unigram_frequencies.iteritems():
        frequency_distribution[genre] = Counter(word_counts.values())
    
    #Applying good turing smoothing on frequencies less than n
    smoothed_frequencies = defaultdict(dict)
    for genre, count_freq in frequency_distribution.iteritems():
        for i in range(min(count_freq.keys()),n):
            smoothed_frequencies[genre][i] = (i + 1) * (float(count_freq[i+1])/count_freq[i])
            
    
    #Updating the counts which were less than n in the original frequency distribution
    #with the new smoothed counts
    for genre, word_counts in unigram_frequencies.iteritems():
        for token, count in word_counts.iteritems():
            if count < n:
                word_counts[token] = smoothed_frequencies[genre][count]
    
    return unigram_frequencies      