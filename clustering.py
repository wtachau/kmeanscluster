from sklearn.cluster import KMeans, Ward
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale, normalize

from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
import numpy as np
import pdb

csv = 'semeion.csv'

dataset = np.genfromtxt(open(csv), delimiter=',')
dataset = scale(dataset)



#  Use 10 clusters.  (k=10)
kmeans = KMeans(n_clusters=10, n_init=10)
#ward = Ward(10)

target = np.array([x[-1] for x in dataset])
train = scale(np.array([x[:-1] for x in dataset]))

kmeans.fit(train)

#kmeans.fit(train)

=======
# initialize cross validation subsets
cv = KFold(len(train), n_folds=4)

correct = sum(kmeans.labels_[i] == target[i] for i in range(len(train)))

#  Train all exs at once
#kmeans = KMeans(10)
#kmeans.fit([train[traincv] for 

"""
for traincv, testcv in cv:
<<<<<<< HEAD

    kmeans = KMeans(n_clusters=10, n_init=100, init='k-means++')
    
    #  Fit kmeans learner on training subset.
    kmeans.fit(train[traincv])
    #kmeans.fit([train[i] for i in traincv])
    
    #  Test the learner using testing subset.
    correct = 0
    num_exs = len(testcv)
    c = sum(target[i] == kmeans.predict(train[i]) for i in testcv)
    print c, num_exs
"""

	kmeans = KMeans(10).fit(train[traincv])

	num_correct = 0
	for index in testcv:
		test_example = train[index]
		target_example = target[index]
		prediction = kmeans.predict(test_example)
		if prediction == target_example:
			num_correct += 1
		#pdb.set_trace()

	print ">>: " + str(num_correct)
	


