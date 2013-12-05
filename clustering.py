from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
import numpy as np

csv = '/Users/SDA/Fall 2013/CS158/kmeanscluster/semeion.csv'

dataset = np.genfromtxt(open(csv), delimiter=',')

#  Use 10 clusters.  (k=10)
kmeans = KMeans(10)

target = np.array([x[-1] for x in dataset])
train = np.array([x[:-1] for x in dataset])

kmeans.fit(train)

cv = KFold(len(train), n_folds=4)


#  Train all exs at once
#kmeans = KMeans(10)
#kmeans.fit([train[traincv] for 

for traincv, testcv in cv:
    print len(traincv)
    print traincv
    print testcv
    #kmeans = KMeans(10).fit(train[traincv])

     #   print kmeans.predict(train[ex])
      #  print target[traincv]