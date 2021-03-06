'''
Created on Sep 11, 2015

@author: gaurav
'''

from utils.ModelingUtilities  import serializeModelToDisk, genres, training_path, getTokensForFile
from nltk.tokenize            import word_tokenize
from nltk.data                import load
from collections              import defaultdict, Counter
from Smoothing.GoodTuring     import applyGoodTuringBigramSmoothing
from matplotlib.font_manager  import path
import os

def generateBigramModels( random_sentence = False ):
    '''
        Controller module for the generation of the bigram models. Calls the various
        methods needed to generate the model and serialise it to the disc.
    '''
    bigrams              = {}
    startchar_successors = {}
    
    # Get the bigrams in the corpus by the genre and the list of tokens that are
    # sentence starter words in the corpus
    for genre in genres:
        print("\nReading files for genre {0}".format(genre))
        path = training_path + genre
        bigrams[genre], startchar_successors[genre] = getBigramsForGenre(path)
        
    #Creating the frequency model of the bigrams
    bigram_frequencies = getBigramFrequencies(bigrams)
    
    #Adding the frequency of the bigrams that include the start character
    if random_sentence:
        bigram_frequencies_with_startChar = getStartCharBigramFrequencies(bigram_frequencies, startchar_successors)
        bigram_model = createBigramModel(bigram_frequencies_with_startChar)
        serializeModelToDisk(bigram_model, 'BigramSentenceModel')
    
    else:
        bigram_model = createBigramModel(bigram_frequencies)
        serializeModelToDisk(bigram_model, 'Bigram')
    

    return bigram_model

def getBigramsForGenre(dir_path, unknown_words = True):
    '''
        Reads through the contents of a complete directory path and finds
        the bigrams present in a genre level corpus
    '''
    genre_tokens               = []
    genre_startchar_successors = []
    
    for path in os.listdir(dir_path):
        
        #Reading the file's contents and getting the tokens
        if not path.startswith('.'):
            tokens = getTokensForFile(dir_path + '/' + path)
            genre_tokens.extend(tokens)

    #Finding the list of words that are sentence starters in the current corpus
    genre_startchar_successors.extend(getStartCharSuccessorsForGenre(genre_tokens))

    #Modifying the list of tokens by inserting <UNKNOWN> for tokens that occur only once
    mod_tokens = insertUnknownWords(genre_tokens)

    #Create a list of bigrams from the tokens
    #Will include the bigrams spanning the end of one file to the beginning of the next.
    # but that doesn't really matter (~5 bigrams out of 10s of thousands)
    genre_bigram = [(mod_tokens[i], mod_tokens[i+1]) for i in range(0, len(mod_tokens)-1)]

    return genre_bigram, genre_startchar_successors

def insertUnknownWords(tokens):
    '''
        Inserts <UNKNOWN> for tokens that occur only once
    '''
    #Creating a list of tokens that need to be replaced since their frequency is 1
    frequencies = Counter(tokens)
    tokens_to_replace = {key for key, value in frequencies.iteritems() if value == 1}
        
    #Modifying the list to have <UNKOWN> for all tokens with frequency = 1  
    mod_tokens = ['<UNKNOWN>' if current_token in tokens_to_replace else current_token for current_token in tokens]

    return mod_tokens
    
def getStartCharSuccessorsForGenre(content):
    '''
        Uses nltk's sentence detector to determine where to place STARTCHAR's,
        and returns all tokens succeeding these STARTCHAR's. 
    '''

    #Train a sentence-subdivision model to be used to identify where to place STARTCHAR characters
    sent_detector = load('tokenizers/punkt/english.pickle')

    #Splitting the corpus into sentences using nltk's tokenizer
    sentences_split_corpus = sent_detector.tokenize(' '.join(content).strip())
        
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

    # Smooth bigram counts
    smoothed_simple_frequency_distribution = applyGoodTuringBigramSmoothing(simple_frequency_distribution, n=5)
    
    #Creates a advanced bigram frequency model distribution like { a : { b:2, c:3 } }
    advanced_bigram_model = createAdvancedBigramFrequency(smoothed_simple_frequency_distribution, bigrams )
    
    return advanced_bigram_model

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
        vocab_size = len(bigram_frequencies[genre]) # approximately
        for word, followers in bigram_frequencies[genre].iteritems():
            # Different words have different number of <UNSEEN> tokens following them
            # The number of such tokens depends upon the number of *seen* words they have following them
            # The equation is:  |w_(i-1)| + [V - w_(i-1)-followers] * |<UNSEEN>|
            seen_word_count = sum(followers.values()) - followers.get('<UNSEEN>',0)
            unseen_word_count = (vocab_size - len(followers) + 1) * followers.get('<UNSEEN>',0)
            total_count = float(seen_word_count + unseen_word_count)
            
            for following_word, count in followers.iteritems():
                genre_level_model[word].update({following_word : count/total_count})
        
        bigram_model[genre] = genre_level_model                    
    
    print("Finished creating the bigram model")
    return bigram_model            
