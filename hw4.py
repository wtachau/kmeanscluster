from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import cross_validation, preprocessing, decomposition, svm
import logloss
import pdb
import numpy as np
import pandas as pd
import pylab as pl

def main():
    """ GET THE DATA """
    #read in  data, parse into training and target sets
    #dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:] 

    #for now, use small dataset for testing
    print "reading data..."
    dataset = np.genfromtxt(open('sample_data/train_sample10000.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])

    """ INITIALIZE CROSS VALIDATOR """
    #Simple K-Fold cross validation. 5 folds.
    print "initializing cross validation..."
    cv = cross_validation.KFold(len(train), n_folds=5, indices=False)


    """ CROSS VALIDATION FOR K NEAREST NEIGHBORS """
    # Run cross validation for KNN: change if statement to True to run
    if (True):
        print "Running cross validation for KNN..."
        knn_results = []
        for k in range(1,10,1):

            # KNN
            knn = KNeighborsClassifier(n_neighbors=k)

            num_correct = 0.0
            num_total = 0.0
            #iterate through the training and test cross validation segments
            for traincv, testcv in cv:
                # FIT THE LEARNER YOU ARE TESTING
                probas = knn.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
                for index, x in enumerate(probas):
                    highest_prob = np.where(x==max(x))[0][0]
                    if target[testcv][index] == highest_prob:
                        num_correct+=1
                    num_total+=1

            #print out the mean of the cross-validated results
            accuracy = num_correct/num_total
            print "Results for KNN w/ k = %s: %s" % (k, accuracy)
            knn_results.append([k, accuracy])

        knn_results = pd.DataFrame(knn_results, columns=["k", "accuracy"])
        pl.plot(knn_results.k, knn_results.accuracy)
        pl.title("Accuracy with Increasing K")
        pl.show()



    """ CROSS VALIDATION FOR SVM with PCA"""
    # Run cross validation for SVM: change if statement to True to run
    if (False):
        print "Running cross validation for SVM with PCA..."

        print "creating, training pca..."
        pca = decomposition.RandomizedPCA(whiten=True)
        pca.fit(train)

        print "RE-initializing cross validation for SVM with PCA..."
        train_transform = pca.transform(train)
        train_scaled = preprocessing.scale(train_transform)
        cv_scaled = cross_validation.KFold(len(train_scaled), n_folds=5, indices=False)

        #Initialize SVM learner
        svm = SVC(C=5., probability=True)

        svm_results = []
        num_correct = 0.0
        num_total = 0.0
        for traincv, testcv in cv_scaled:
            # FIT THE LEARNER YOU ARE TESTING
            probas = svm.fit(train_scaled[traincv], target[traincv]).predict_proba(train_scaled[testcv])
            for index, x in enumerate(probas):
                highest_prob = np.where(x==max(x))[0][0]
                if target[testcv][index] == highest_prob:
                    num_correct+=1
                num_total+=1

        #print out the mean of the cross-validated results
        accuracy = num_correct/num_total
        print "Results for SVM: %s" %(accuracy)
if __name__=="__main__":
    main()
