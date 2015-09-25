'''
Created on Sep 25, 2015

@author: gaurav
'''

from Smoothing.Perplexity import getUnigramPerplexity, getBigramPerplexity

def classifyBooksWithUnigram():
    '''
        Find the perplexities of the test books using the bigram model and classify
        the book into the respective genres using these value
    '''
    #Calculate the perpexities of the test books using the unigram model
    perplexity_test_books  = getUnigramPerplexity()
    print("\n Classifying books with the unigram model")
    
    #The genre of a book is the model with which it produced the lowest perplexity
    for book, perplexities in perplexity_test_books.iteritems():
        genre = [key for key, value in perplexities.iteritems() if value == min(perplexities.values())][0]
        print("Genre of {0} is {1}".format(book, genre))
        
def classifyBooksWithBigram():
    '''
        Find the perplexities of the test books using the bigram model and classify
        the book into the respective genres using these value
    '''
    #Calculate the perpexities of the test books using the bigram model
    perplexity_test_books  = getBigramPerplexity()
    print("\n Classifying books with the bigram model")
    
    #The genre of a book is the model with which it produced the lowest perplexity
    for book, perplexities in perplexity_test_books.iteritems():
        genre = [key for key, value in perplexities.iteritems() if value == min(perplexities.values())][0]
        print("Genre of {0} is {1}".format(book, genre))

if __name__ == '__main__':
    classifyBooksWithUnigram()
    classifyBooksWithBigram()