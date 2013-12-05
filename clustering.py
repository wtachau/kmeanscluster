import sklearn.cluster
import numpy as np

csv = '/Users/SDA/Fall 2013/CS158/semeion.csv'

dataset = np.genfromtxt(open(csv), delimiter=',')

#  Use 10 clusters.  (k=10)
kmeans = sklearn.cluster.KMeans(10)

target = np.array([x[-1] for x in dataset])
train = np.array([x[:-1] for x in dataset])

kmeans.fit(train)