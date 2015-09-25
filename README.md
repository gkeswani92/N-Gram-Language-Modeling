# N-Gram-Language-Modeling

We are writing a program that computes unsmoothed unigrams and bigrams for an arbitrary text corpus, in this case
open source books from Gutenberg.org. Since we are working with raw texts, so we need to do tokenization, based on 
the design decisions we make.


(a) Data and preprocessing: We are working with a selection of books from gutenberg.org which are separated according
                            to their genres: children’s books, crime/detective and history/biography. We will use the
                            books as our corpora to train language models.

(b) Random sentence generation: We will generate random sentences based on the unigram and bigram 
                                language model that we generated in the first step. We will also do the same with seeding, 
                                i.e. starting from an incomplete sentence of our choice and completing it by 
                                generating from our language model, instead of generating from scratch.
                                
(c) Smoothing: We will implement Add-One smoothing and Good-Turing smoothing to improve the accuracy of the language
               models

(d) Handle unknown words: We will implement an algorithm that will allow us to handle unknown words in the test data
                          that were not found in the training data.
                          
(e) Genre Classification: Using K-Nearest Neighbor algorithm to classify test books into one of the three genres - 
						  crime, history or children
                          
