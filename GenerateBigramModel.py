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
    bigram_frequencies = getBigramFrequencies(bigrams, simple = True, use_nltk = False)
    
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

def getBigramFrequencies( bigrams, simple = True, use_nltk = False ):
    '''
        Computes the frequencies of the bigrams in two ways depending on the 
        value of the param simple.
        
        If simple is True, it returns a bigram frequency model like { (a,b) : 2, (a,c) : 3 }
        If simple is False, it returns a bigram frequency model like { a : { b:2, c:3 } }
                            in which the calculations become much faster
    '''
    print("\nComputing the frequency of the bigrams")
    
    #If we want a bigram frequency model like { (a,b) : 2, (a,c) : 3 }
    if simple:
        return createSimpleBigramFrequency(bigrams) 
        
    #If we want a bigram frequency model like { a : { b:2, c:3 } }
    else:
        return createAdvancedBigramFrequency(bigrams, use_nltk)
            
def createSimpleBigramFrequency( bigrams ):
    '''
        Creates bigram frequency model like { (a,b) : 2, (a,c) : 3 }
    '''
    bigram_frequencies = {}
    
    for genre, pairs in bigrams.items():    
        bigram_frequencies[genre] = Counter(pairs)
    
    print("Finished computing the frequency of the bigrams")
    return bigram_frequencies

def createAdvancedBigramFrequency( bigrams, use_nltk ):
    '''
        Creates bigram frquency model like { a : { b:2, c:3 } }
        in which the calculations become much faster
    '''
    bigram_frequencies = {}
    
    for genre, pairs in bigrams.items():
        
        #Using NLTK if flag is passed through which is not by default
        if use_nltk:
            #NLTK package conditional probabilities. Storing it like this would be ideal
            bigram_frequencies[genre] = nltk.ConditionalFreqDist(bigrams['children'])
        
        #Manual method to create the frequency distribution
        else: 
            genre_bigram_frequency = defaultdict(lambda: defaultdict(dict))
            
            #Keep updating the dictionary of the first part of the bigram 
            #with the second word and the number of times it appeared
            for bigram in pairs:
                if not bigram[1] in genre_bigram_frequency[genre][bigram[0]]:
                    genre_bigram_frequency[genre][bigram[0]].update({bigram[1] : pairs.count(bigram)})
            
            bigram_frequencies[genre] = genre_bigram_frequency
                
    print("Finished computing the frequency of the bigrams")
    return bigram_frequencies

#Temportary call to the main bigram model method
generateBigramModels()
                
   
