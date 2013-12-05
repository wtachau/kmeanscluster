
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
import numpy as np
import pdb

csv = 'semeion.csv'

dataset = np.genfromtxt(open(csv), delimiter=',')
dataset = scale(dataset)


#  Use 10 clusters.  (k=10)
kmeans = KMeans(n_clusters=10)

target = np.array([x[-1] for x in dataset])
train = np.array([x[:-1] for x in dataset])

kmeans.fit(train)

# initialize cross validation subsets
cv = KFold(len(train), n_folds=4)


#  Train all exs at once
#kmeans = KMeans(10)
#kmeans.fit([train[traincv] for 

for traincv, testcv in cv:
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
	


