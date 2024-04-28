import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics 

# kFold: Designates the number of folds we will be using
kFold = 5
# svmTypes: Stores the kernel functions we will be working with
svmTypes = ['linear', 'poly', 'rbf']

'''
Function: Function that performs SVM and trains/tests our dataset
Parameters: Training dataset's features/labels and testing dataset's features/labels
Return Type: Array
'''
def performSVM(trainPixels, testPixels, trainLabels, testLabels):
    # We will have an array that stores each kernel function's accuracy
    funcAcc = []
    # We will loop through each kernel function 
    for kFunc in svmTypes:
        # First, we create our SVM with the current kernel we are using
        svm_MNIST = SVC(kernel=kFunc)
        # Then we train our SVM with the training data
        svm_MNIST.fit(trainPixels, trainLabels)
        # We then predict what labels we will get from our SVM
        predictLabels = svm_MNIST.predict(testPixels)
        # We finally compute the accuracy of our prediction and save the result
        # to the accuracy array
        funcAcc.append(metrics.accuracy_score(testLabels, predictLabels))
    # We then return the accuracy array to main
    return funcAcc

'''
Description: Function that prints out the accuracy of each fold
Parameters: Header string and the array containing each fold's accuracy
Return Type: None
'''
def printAcc(header, accArray):
    # Print the header
    print(f"\n\n{header}\n")
    # Print out each fold's accuracy
    for i in range(len(accArray)):
        print(f"\tFold {i+1} Accuracy: {accArray[i]*100:.2f}%")
    # Print out the average accuracy for each kernel function used
    print(F"\n\tAverage Accuracy: {np.mean(accArray)*100:.2f}%\n")

'''
Description: Main function of program
Parameters: None
Return Type: None
'''
def main():
    # First, we read in the data from the csv file
    dataMNIST = pd.read_csv("MNIST.csv")

    # Then, we will celebrate our features from our labels
    labels = dataMNIST.iloc[:,0]
    pixels = dataMNIST.iloc[:,1:]

    # We now set up the folds we will be using while using SVM
    folds = KFold(n_splits=kFold)
    # We also set a counter to keep track of the current fold
    currFold = 0

    # Finally, we set up three arrays that will store each kernel
    # function's accuracies 
    linearAcc = np.zeros(kFold)
    polyAcc = np.zeros(kFold)
    rbfAcc = np.zeros(kFold)

    for train, test in folds.split(pixels):
        # For each fold, we will separate the training and testing 
        # features and labels from each other
        trainPixels = pixels.iloc[train, :]
        testPixels = pixels.iloc[test, :]

        trainLabels = labels.iloc[train]
        testLabels = labels.iloc[test]

        # Then, we call performSVM that will compute a SVM 
        accLst = performSVM(trainPixels, testPixels, trainLabels, testLabels)

        # We now input each SVM kernel function's accuracies in their
        # respective arrays and then move on to the next fold
        linearAcc[currFold] = accLst[0]       
        polyAcc[currFold] = accLst[1]            
        rbfAcc[currFold] = accLst[2] 
        currFold += 1           
    
    # We end the program by printing out each kernel function's 
    # accuracies for each fold and the average accuracy for each one
    printAcc("Linear Kernel Accuracy", linearAcc)
    printAcc("Polynomial Kernel Accuracy", polyAcc)
    printAcc("RBF Kernel Accuracy", rbfAcc)

if __name__ == "__main__":
    main()
