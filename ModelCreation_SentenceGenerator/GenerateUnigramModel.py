'''
Created on Sep 9, 2015

@author: gaurav
'''

from utils.ModelingUtilities  import genres, training_path, serializeModelToDisk, getTokensForFile, training_file_counts
from Smoothing.GoodTuring     import applyGoodTuringUnigramSmoothing
from collections              import Counter, defaultdict
import os
from numpy.random             import choice
import csv

def generateUnigramModels():
    '''
        ControllerModelAndRandomSentence for the generation of the unigram models. 
        Iterates over the given genre folders and retrieves the unigram model
        to create the final unigram model dictionary
    '''
    #Get the frequency of each word in the corpus
    unigram_frequencies = getUnigramFrequenciesforTrainingSet()
                
    #Get the word type and token count for the corpus
    unigram_features = getUnigramModelFeatures(unigram_frequencies)
    print("\n Unigram Features (Word Types, Work Tokens) {0} \n".format(unigram_features))
    
    #Returns the frequency distributions with all tokens with frequency 1 replacedby <UNKNOWN>
    #unigram_features_unknown_words = handleUnknownWords(unigram_frequencies)
    unknown_word_probs = getUnknownWordSamplingProbs(unigram_frequencies)
    #unigram_frequencies = {} #Releasing the unused memory to speed up the program
    
    #Performing Good Turing Smoothing
    #smoothed_frequencies = applyGoodTuringUnigramSmoothing(unigram_features_unknown_words, n = 5)
    
    #Creating the unigram model i.e. calculating the probabilities of the unigrams
    #unigram_model = createUnigramModel(smoothed_frequencies, unigram_features)
    #unigram_model = createUnigramModel(unigram_features_unknown_words, unigram_features)
    unigram_model = createUnigramModel(unigram_frequencies, unigram_features, unknown_word_probs)
     
    #Storing the model on the disk in JSON format
    serializeModelToDisk(unigram_model, 'UnigramSampled')
    
    return unigram_model

def getUnigramFrequenciesforTrainingSet():
    '''
        Wrapper method to get the unigram frequency distribution across all 
        genres
    '''
    unigram_frequencies = {}
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        word_list = []
        
        #Reads in the unigrams one file at a time
        for path in os.listdir(training_path + genre):
            word_list.extend(getTokensForFile(training_path + genre + '/' + path))
        
        #Creating a counter of the frequencies at the genre level
        unigram_frequencies[genre] = Counter(word_list)
    
    return unigram_frequencies
        
def getUnigramModelFeatures(unigram_models):
    '''
        Creates a dictionary of genre : (word_types, word_tokens) where word
        types are the unique words while work tokens is the total count of the 
        words in the corpus.
    '''
    unigram_features = {}
    for genre, model in unigram_models.items():
        word_types  = len(model.keys())
        word_tokens = sum(model.values())
        unigram_features[genre] = (word_types, word_tokens)
        
    return unigram_features

def handleUnknownWords(unigram_features):
    '''
        Replaces all tokens with the frequency 1 as <UNKNOWN> and
        aggregates them into one entry at the genre level
    '''
    modified_unigram_frequency = {}
    for genre, frequency_dist in unigram_features.items():
        count = 0
        
        for token, frequency in frequency_dist.items():
            if frequency == 1:
                del frequency_dist[token]
                count += 1
        
        frequency_dist['<UNKNOWN>'] = count
        modified_unigram_frequency[genre] = frequency_dist
    
    return modified_unigram_frequency

def getUnknownWordSamplingProbs(unigram_features, iterations=100):
    '''
        Samples n_tokens/'training_file_counts' number of tokens from
        each of the training corpora 'iterations' many times, w/o replacement.

        This mimics the behavior of drawing a random text, which is the size of
        the average book among the training books, and seeing how many of the words
        in that "new" text are novel. This provides a good estimation for the <UNKOWN> frequency.
    '''
    unknown_frequency_counts = defaultdict(list)
    unknown_frequency_probs = {}
    for genre, frequency_dist in unigram_features.items():
        # Convert Counter back into list
        token_list = [key for key,value in frequency_dist.items() for _ in range(value)]
        sample_size = len(token_list)/training_file_counts[genre]
        for i in range(iterations):
            new_tokens_counter = Counter(choice(token_list,sample_size,replace=False))
            old_tokens_counter = frequency_dist - new_tokens_counter
            novel_tokens_counter = new_tokens_counter - old_tokens_counter
            unknown_frequency_counts[genre].append(sum(novel_tokens_counter.values()))
            #novel_words_set = set(new_words_counter.keys()) - set(old_words_counter.keys())
            #unknown_frequency_counts[genre].append(len(novel_words_set))
        # Compute average probabilities from unknown counts
        average_count = sum(unknown_frequency_counts[genre]) * 1.0/iterations
        unknown_frequency_probs.update({genre: average_count * 1.0/sample_size})

    # Write the unknown counts/probs to a file
    dir_path = '/Users/Macbook/Documents/Cornell/CS 4740 - Natural Language Processing/Project 1/N-Gram-Language-Modeling/'
    w = csv.writer(open(dir_path+'sampling_counts_dump.csv', "w"))
    for key, val in unknown_frequency_counts.items():
        w.writerow([key]+val)
        #w.writerow([key, val])
    w = csv.writer(open(dir_path+'sampling_probs_dump.csv', "w"))
    for key, val in unknown_frequency_probs.items():
        w.writerow([key, val])

    return unknown_frequency_probs

def createUnigramModel(unigram_frequencies, unigram_features, unknown_word_probs):
    '''
        Uses the word frequency and the work token count to create the 
        unigram model per genre i.e genre : { word1 : probability1, word2 : 
        probability2 }
    '''
    unigram_model = {}
    
    for genre, frequencies in unigram_frequencies.items():
        
        #Number of tokens for the particular genre is used to calculate probabilities of a word occuring in that genre
        token_count = unigram_features[genre][1]

        # Probability mass remaining after adding '<UNKNOWN>' word type
        # Need this additional factor to ensure that everything sums to zero
        remaining_prob_mass = 1 - unknown_word_probs[genre]

        #Creating the model at the genre level
        unigram_model[genre] = dict((word, remaining_prob_mass * frequency * 1.0 / token_count) for word, frequency in frequencies.items())
        unigram_model[genre].update({'<UNKNOWN>': unknown_word_probs[genre]})

    return unigram_model

        
