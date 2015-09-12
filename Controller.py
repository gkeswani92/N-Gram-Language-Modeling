'''
Created on Sep 10, 2015

@author: gaurav
'''

from GenerateUnigramModel import generateUnigramModels

def formatWordForQuality(word):
    '''
        Replaces the special characters in the passed word with None. This is
        done to weed out the unwanted characters
    '''
    
    #Converting UTF to string
    word = str(word)
    
    #Removing the special characters from the words
    special_characters = '[~!@#$%^?&*()_,.+{}":;/\']+$123456789'
    word = word.translate(None, special_characters)
    
    #Regex method to do the same
    #word = re.sub(special_characters, '', word)
    
    #Converting everything to lower case
    word = word.lower()
    
    return word
    
def main():
    
    #Generate the unigram model for all the genres
    unigram_model = generateUnigramModels()

  
if __name__ == '__main__':
    main()