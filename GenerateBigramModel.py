'''
Created on Sep 11, 2015

@author: gaurav
'''

from ModelingUtilities import serializeUnigramModelToDisk, genres, training_path
from nltk.tokenize     import word_tokenize
from collections       import defaultdict, Counter
import nltk
import os
import codecs

def generateBigramModels():
    '''
        Controller for the generation of the unigram models. 
        Iterates over the given genre folders and retrieves the unigram model
        to create the final unigram model dictionary
    '''
    #Get the bigrams in the corpus by the genre
    bigrams = {}
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        bigrams[genre] = getBigramsForGenre(training_path+genre)
        
    #Creating the frequency model of the bigrams
    bigram_frequencies = getBigramFrequencies(bigrams, simple = False, use_nltk = False)
    
    #Creating the bigram model i.e. calculating the probabilities of the unigrams
    bigram_model = createBigramModel(bigram_frequencies)
    
    #Storing the model on the disk in JSON format
    serializeUnigramModelToDisk(bigram_frequencies, 'Bigram')

def getBigramsForGenre(dir_path):
    '''
        Reads through the contents of a complete directory path and finds
        the bigrams present in a genre level corpus
    '''
    content = []
    genre_bigram = []
    
    for path in os.listdir(dir_path):
        
        #Reading the file's contents
        file_path = dir_path + '/' + path
        print("Reading file at {0}".format(file_path))
        
        #Creating a list of all the tokens in the file
        f = codecs.open(file_path,'r','utf8', errors='ignore')
        content = word_tokenize(f.read())
        f.close()
        
        #All books in the same genre are aggregated into one bigram list
        #Keeping the use_nltk param false by default, so that is used 
        #only when we want to use it explicitly
        genre_bigram.extend(createBigrams(content, use_nltk = False))
    
    return genre_bigram

def createBigrams( content, use_nltk = False ):
    '''
        Creates pair of bigrams in a tuple (word1, word2) and adds them all to a list
    '''
    #This is not used by default and is present only for testing purposes
    if use_nltk:
        return list(nltk.bigrams(content))
        
    #Manual method to create a list of bigrams from the given content
    bigram_list = [(content[i], content[i+1]) for i in range(0, len(content)-1)]
    return bigram_list

def getBigramFrequencies( bigrams, simple = False, use_nltk = False ):
    '''
        Computes the frequencies of the bigrams in two ways depending on the 
        value of the param simple.
        
        If simple is True, it returns a bigram frequency model like { (a,b) : 2, (a,c) : 3 }
        If simple is False, it returns a bigram frequency model like { a : { b:2, c:3 } }
                            in which the calculations become much faster
    '''
    print("\nComputing the frequency of the bigrams")
    
    simple_frequency_distribution = createSimpleBigramFrequency(bigrams) 
    
    if simple:
        return simple_frequency_distribution
        
    #If we want a bigram frequency model like { a : { b:2, c:3 } }
    else:
        return createAdvancedBigramFrequency(simple_frequency_distribution, bigrams, use_nltk)
            
def createSimpleBigramFrequency( bigrams ):
    '''
        Creates bigram frequency model like { (a,b) : 2, (a,c) : 3 }
    '''
    bigram_frequencies = {}
    
    for genre, pairs in bigrams.items():    
        bigram_frequencies[genre] = Counter(pairs)
    
    print("Finished computing the simple frequency of the bigrams")
    return bigram_frequencies

def createAdvancedBigramFrequency( simple_frequency_distribution, bigrams, use_nltk ):
    '''
        Creates bigram frquency model like { a : { b:2, c:3 } }
        in which the calculations become much faster
    '''
    bigram_frequencies = {}
    
    for genre in genres:
        
        #Using NLTK if flag is passed through which is not by default
        if use_nltk:
            bigram_frequencies[genre] = nltk.ConditionalFreqDist(bigrams['children'])
        
        #Manual method to create the frequency distribution
        else: 
            genre_level_frequencies = defaultdict(dict)
            
            for bigram, count in simple_frequency_distribution[genre].iteritems():
                genre_level_frequencies[bigram[0]].update({bigram[1]:count})
                
            bigram_frequencies[genre] = genre_level_frequencies
                
                
    print("Finished computing the advanced frequency of the bigrams")
    return bigram_frequencies
                
def createBigramModel( bigram_frequencies ): 
    '''
        Creating the bigram model for words depending on which kind of frequency
        model was passed to it
    '''
    print("\nCreating the bigram model")
    bigram_model = {}
    
    for genre in genres:
        genre_level_model = defaultdict(dict)
        
        for word, followers in bigram_frequencies[genre].iteritems():
            total_count = float(sum(followers.values()))
            
            for following_word, count in followers.iteritems():
                genre_level_model[word].update({following_word : count/total_count})
        
        bigram_model[genre] = genre_level_model                    
    
    print("Finished creating the bigram model")
    return bigram_model            
         
