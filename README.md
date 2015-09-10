# N-Gram-Language-Modeling

We are writing a program that computes unsmoothed unigrams and bigrams for an arbitrary text corpus, in this case
open source books from Gutenberg.org. Since we are working with raw texts, so we need to do tokenization, based on 
the design decisions we make.


(a) Data and preprocessing: We are working with a selection of books from gutenberg.org which are separated according
                            to their genres: children’s books, crime/detective and history/biography. We will use the
                            books as our corpora to train your language models.

(b) Random sentence generation: We will write code for generating random sentences based on a unigram or bigram 
                                language model that we generated in the first step. We will also with seeding, 
                                i.e., starting from an incomplete sentence of our choice and completing it by 
                                generating from our language model, instead of generating from scratch
                                
(c) Smoothing: We will implement Add-One smoothing Good-Turing smoothing to improve the accuracy of the language
               models

(d) Handle unknown words: We will implement an algorithm that will allow us to handle unknown words in the test data
                          that were not found in the training data.
                          
(e) Genre Classification: In this part of the project, we will determine a method that uses our language models 
                          to classify books from the test sets according to their genres
                          
