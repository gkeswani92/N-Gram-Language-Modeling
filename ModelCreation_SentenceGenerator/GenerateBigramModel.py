'''
Created on Sep 11, 2015

@author: gaurav
'''

from ModelCreation_SentenceGenerator.ModelingUtilities  import serializeModelToDisk, genres, training_path
from nltk.tokenize                                      import word_tokenize
from nltk.data                                          import load
from collections                                        import defaultdict, Counter
import os
import codecs

def generateBigramModels():
    '''
        ControllerModelAndRandomSentence for the generation of the bigram models. Calls the various
        methods needed to generate the model and serialise it to the disc.
    '''
    bigrams              = {}
    startchar_successors = {}
    
    #Get the bigrams in the corpus by the genre and the list of tokens that are 
    #sentence starter words in the corpus
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        bigrams[genre], startchar_successors[genre] = getBigramsForGenre(training_path+genre)
        
    #Creating the frequency model of the bigrams
    bigram_frequencies = getBigramFrequencies(bigrams)
    
    #Adding the frequency of the bigrams that include the start character
    bigram_frequencies_with_startChar = getStartCharBigramFrequencies(bigram_frequencies, startchar_successors)

    #TODO: Have only refactored code until this point. 
    #Creating the bigram model i.e. calculating the probabilities of the unigrams
    bigram_model = createBigramModel(bigram_frequencies_with_startChar)

    #Storing the model on the disk in JSON format
    serializeModelToDisk(bigram_model, 'Bigram')

    return bigram_model

def getBigramsForGenre(dir_path):
    '''
        Reads through the contents of a complete directory path and finds
        the bigrams present in a genre level corpus
    '''
    genre_bigram         = []
    genre_startchar_successors = []
    
    for path in os.listdir(dir_path):
        
        #Reading the file's contents
        file_path = dir_path + '/' + path
        print("Reading file at {0}".format(file_path))
        f = codecs.open(file_path,'r','utf8', errors='ignore')
        corpus = f.read()
        f.close()
        
        #Creating a list of all the tokens in the file using nltk's tokenize method
        tokens = word_tokenize(corpus);
        bigrams = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens)-1)]
        genre_bigram.extend(bigrams)
        
        #Finding the list of words that are sentence starters in the current corpus
        genre_startchar_successors.extend(getStartCharSuccessorsForGenre(corpus))
    
    return genre_bigram, genre_startchar_successors

def getStartCharSuccessorsForGenre(content):
    '''
        Uses nltk's sentence detector to determine where to place STARTCHAR's,
        and returns all tokens succeeding these STARTCHAR's. 
    '''

    #Train a sentence-subdivision model to be used to identify where to place STARTCHAR characters
    sent_detector = load('tokenizers/punkt/english.pickle')

    #Splitting the corpus into sentences using nltk's tokenizer
    sentences_split_corpus = sent_detector.tokenize(content.strip())
        
    #Extracting the first word of every sentence and storing it as a sentence starter
    sentence_start_words = [word_tokenize(sentence)[0] for sentence in sentences_split_corpus]
    
    return sentence_start_words

def getBigramFrequencies( bigrams ):
    '''
        Computes a simple frequency of the bigrams and then use it to compute the
        more advanced version of the frequency distribution
    '''
    print("\nComputing the frequency of the bigrams")
    
    #Creates a simple bigram frequency model distribution like { (a,b) : 2, (a,c) : 3 }
    simple_frequency_distribution = dict((genre, Counter(pairs)) for genre, pairs in bigrams.iteritems())
    
    #Creates a advanced bigram frequency model distribution like { a : { b:2, c:3 } }
    return createAdvancedBigramFrequency(simple_frequency_distribution, bigrams)

def createAdvancedBigramFrequency( simple_frequency_distribution, bigrams ):
    '''
        Creates bigram frquency model like { a : { b:2, c:3 } } in which the calculations 
        become much faster than the simple frequency distribution model
    '''
    bigram_frequencies = {}
    
    for genre in genres:
        genre_level_frequencies = defaultdict(dict)
        
        for bigram, count in simple_frequency_distribution[genre].iteritems():
            genre_level_frequencies[bigram[0]].update({bigram[1]:count})
            
        bigram_frequencies[genre] = genre_level_frequencies
                
    print("Finished computing the advanced frequency of the bigrams")
    return bigram_frequencies

def getStartCharBigramFrequencies(bigram_frequencies, startchar_successors):
    '''
        Add the frequencies of the bigrams involving the start character
    '''
    for genre in genres:
        bigram_frequencies[genre]['<START>'].update(Counter(startchar_successors[genre]))
        
    return bigram_frequencies

def createBigramModel( bigram_frequencies ): 
    '''
        Creating the bigram model for words
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
