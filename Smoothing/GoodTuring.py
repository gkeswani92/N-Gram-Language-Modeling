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

def applyGoodTuringBigramSmoothing(bigram_frequencies, n = 5):
    '''
        Takes the frequency distribution as an input and returns the smoothed
        frequency distribution
    '''
    #Counter to determine how many unigrams occur 0 times, 1 times, 2 times ...
    frequency_distribution = {}
    total_bigram_count = {}
    unseen_bigram_count = {}
    for genre, bigram_counts in bigram_frequencies.iteritems():
        total_bigram_count[genre] = len({bigram[0] for bigram in bigram_counts.keys()})**2
        unseen_bigram_count[genre] = total_bigram_count[genre] - len(bigram_counts)
        frequency_distribution[genre] = Counter(bigram_counts.values())
        frequency_distribution[genre].update({0 : unseen_bigram_count[genre]})
    
    #Applying good turing smoothing on frequencies less than n
    smoothed_frequencies = defaultdict(dict)
    for genre, count_freq in frequency_distribution.iteritems():
        for i in range(min(count_freq.keys()),n):
            smoothed_frequencies[genre][i] = (i + 1) * (float(count_freq[i+1])/count_freq[i])
            
    
    #Updating the counts which were less than n in the original frequency distribution
    #with the new smoothed counts
    unseen_freqs = {}
    for genre, bigram_counts in bigram_frequencies.iteritems():
        for token, count in bigram_counts.items():
            if count < n:
                bigram_counts[token] = smoothed_frequencies[genre][count]
                
            #Adding the unseen bit as the 2nd part of a bigram only once
            if (token[0],'<UNSEEN>') not in bigram_counts:
                bigram_counts[(token[0],'<UNSEEN>')] = smoothed_frequencies[genre][0]
        
        #Store the unseen frequency at the genre level
        #unseen_freqs[genre] = smoothed_frequencies[genre][0]
        
    return bigram_frequencies, unseen_freqs