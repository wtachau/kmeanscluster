import sklearn.cluster
from sklearn import cross_validation
import numpy as np
import pdb

csv = 'semeion.csv'

dataset = np.genfromtxt(open(csv), delimiter=',')


#  Use 10 clusters.  (k=10)
kmeans = sklearn.cluster.KMeans(10)

target = np.array([x[-1] for x in dataset])
train = np.array([x[:-1] for x in dataset])

# cross validation
cv = cross_validation.KFold(len(train), n_folds=5, indices=False)

num_correct = 0.0
num_total = 0.0
#iterate through the training and test cross validation segments
for traincv, testcv in cv:
    # FIT THE LEARNER YOU ARE TESTING
    probas = kmeans.fit(train[traincv], target[traincv])
    pdb.set_trace()
    for index, x in enumerate(probas):
    	
        highest_prob = np.where(x==max(x))[0][0]
        if target[testcv][index] == highest_prob:
            num_correct+=1
        num_total+=1