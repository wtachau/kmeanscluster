from sklearn.cluster import KMeans, Ward
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale, normalize
import numpy as np

csv = '/Users/SDA/Fall 2013/CS158/kmeanscluster/semeion.csv'

dataset = np.genfromtxt(open(csv), delimiter=',')


#  Use 10 clusters.  (k=10)
kmeans = KMeans(n_clusters=10, n_init=10)
#ward = Ward(10)

target = np.array([x[-1] for x in dataset])
train = scale(np.array([x[:-1] for x in dataset]))

kmeans.fit(train)

#kmeans.fit(train)

cv = KFold(len(train), n_folds=4)

correct = sum(kmeans.labels_[i] == target[i] for i in range(len(train)))

#  Train all exs at once
#kmeans = KMeans(10)
#kmeans.fit([train[traincv] for 

"""
for traincv, testcv in cv:

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