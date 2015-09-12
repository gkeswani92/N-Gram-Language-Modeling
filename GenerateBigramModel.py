'''
Created on Sep 11, 2015

@author: gaurav
'''

from Controller import removeSpecialCharacters
from collections import defaultdict
import nltk
import os
import codecs

#Variable definitions relative to the path of the source files
base_path     = os.path.dirname(__file__)
#genres        = ['children', 'crime', 'history']
genres        = ['history']
training_path = base_path + '/books/train_books/'
test_path     = base_path + '/books/test_books/'

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
    
    #Print this at your own risk :)
    #pprint.pprint(bigrams['children'])
    
    #NLTK package conditional probabilities. Storing it like this would be ideal
    #for key,value in nltk.ConditionalFreqDist(bigrams['children']).items():
    #    print ((key,value))
    
    bigram_frequencies = getBigramFrequencies(bigrams, simple = False)
    print(bigram_frequencies)

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
        f = codecs.open(file_path,'r')
        print("Reading file at {0}".format(file_path))
        
        #Creating a list of all the tokens in the file
        for line in f.readlines():
            line = [x.strip() for x in line.split(' ') if x]
            content.extend(line)
        
        #All books in the same genre are aggregated into one bigram list
        #Keeping the use_nltk param false by default, so that is used 
        #only when we want to use it explicitly
        genre_bigram.extend(createBigrams( content, use_nltk = False ))
        
        break
    
    return genre_bigram

def createBigrams( content, use_nltk = False ):
    '''
        Creates pair of bigrams in a tuple (word1, word2) and adds them all to a list
    '''
    #This is not used by default and is present only for testing purposes
    if use_nltk:
        return list(nltk.bigrams(content))
        
    #Manual method to create a list of bigrams from the given content
    bigram_list = [(removeSpecialCharacters((str(content[i].lower()))), removeSpecialCharacters(str(content[i+1].lower()))) 
                            for i in range(0, len(content)-1) if content[i] and content[i+1]]
    return bigram_list

def getBigramFrequencies( bigrams, simple = True ):
    '''
        Computes the frequencies of the bigrams in two ways depending on the 
        value of the param simple.
        
        If simple is True, it returns a bigram frequency model like { (a,b) : 2, (a,c) : 3 }
        If simple is False, it returns a bigram frequency model like { a : { b:2, c:3 } }
                            in which the calculations become much faster
    '''
    print("\nComputing the frequency of the bigrams")
    bigram_frequencies = {}
    
    #If we want a bigram frequency model like { (a,b) : 2, (a,c) : 3 }
    if simple:
        for genre, pairs in bigrams.items():
            genre_bigram_frequency = defaultdict(int)
            
            #Adding the frequency of each unique bigram to the genre level bigram frequency
            for bigram in pairs:
                if bigram not in genre_bigram_frequency:
                    genre_bigram_frequency[bigram] = pairs.count(bigram)
            
            bigram_frequencies[genre] = genre_bigram_frequency
            
    #If we want a bigram frequency model like { a : { b:2, c:3 } }
    else:
        for genre, pairs in bigrams.items():
            genre_bigram_frequency = defaultdict(lambda: defaultdict(dict))
            
            #Keep updating the dictionary of the first part of the bigram 
            #with the second word and the number of times it appeared
            for bigram in pairs:
                if not bigram[1] in genre_bigram_frequency[genre][bigram[0]]:
                    genre_bigram_frequency[genre][bigram[0]].update({bigram[1] : pairs.count(bigram)})
            
            bigram_frequencies[genre] = genre_bigram_frequency    
    
    print("Finished computing the frequency of the bigrams")
    return bigram_frequencies


     
     
generateBigramModels()
                
   
