'''
Created on Sep 18, 2015

@author: gaurav
'''

from KNearestGenreClassification.KNearestNeighbourClassifier import KNearestNeighbourController
from KNearestGenreClassification.ClassificationUtilities     import getFileListAndLabels
from utils.ModelingUtilities                         import training_path, test_path

def main( validation = True ):
    
    #Creating a list of the training files and the labels for each of the training files
    training_files, training_labels = getFileListAndLabels(training_path)
    
    #If the genre classification validation run is being called. False by Default
    if validation:
        KNearestNeighbourController(training_files, training_labels, k = 1, tf_idf= False)
    
    #If the genre classification is being performed on the test set using K-Nearest Neighbours
    else:
        test_files, test_labels = getFileListAndLabels(test_path)
        KNearestNeighbourController(training_files, training_labels, test_files, test_labels, k = 1, tf_idf = False)
    
if __name__ == '__main__':
    main()