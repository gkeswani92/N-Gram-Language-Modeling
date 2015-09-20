'''
Created on Sep 18, 2015

@author: gaurav
'''

from GenreClassification.KNearestNeighbourClassifier import KNearestNeighbourController
from GenreClassification.ClassificationUtilities import getFileListAndLabels
from ModelCreation_SentenceGenerator.ModelingUtilities import training_path, test_path

def main( validation = True ):
    
    #Creating a list of the training files and the labels for each of the training files
    training_files, training_labels = getFileListAndLabels(training_path)
    
    if validation:
        KNearestNeighbourController(training_files, k = 3)
       
  
if __name__ == '__main__':
    main()