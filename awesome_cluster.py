import numpy as np
import matplotlib.pyplot as plt

from learning import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from collections import defaultdict
from scipy.stats import mode

from kmeansplotting import plot_pca

def semeionData():
    data = DataSet(name="../data/semeion")
    return data


def kmeansLearner(data):
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    train = [row[:-1] for row in data.examples]
    scaled = scale(train)
    kmeans.fit(scaled)

    cluster_label_dict = clusterLabelDict(kmeans, data)

    def predict(ex):
        #print ex
        scaled = scale(ex[:-1])
        cluster = kmeans.predict(scaled)[0]   
        return cluster_label_dict[cluster]
        
    #  Pass `predict` function to cross_validation later.
    return predict


def clusterLabelDict(kmeans, data):
    cluster_label_dict = defaultdict(list)
    
    #  Get arbitrary cluster labels for each prediction.
    for row in data.examples:
        cluster = kmeans.predict(scale(row[:-1]))[0]            
        cluster_label_dict[cluster].append(row[-1])
    
    #  Reset each key to the MODE of each inner list.
    for cluster in cluster_label_dict:
        m = mode(cluster_label_dict[cluster])[0][0]
        cluster_label_dict[cluster] = int(m)
        
    #  Return a normal dict, not a defaultdict<list>.
    return dict(cluster_label_dict)


if __name__ == '__main__':
    pass
    #data = semeionData()
    #results = cross_validation(kmeansLearner, data)
    #print "Kmeans accuracy: ", results
#data = semeionData()
#kmeans = kmeansLearner(data)


def plot_clusters():
    data = semeionData()
    # train = [row[:-1] for row in data.examples]
    # scaled = scale(train)
    plot_pca(data)


def writePBM(ex, filename='example.pbm'):
    """Takes a binary feature vector and writes a PBM file."""
    f = file('/Users/SDA/Fall 2013/CS158/kmeanscluster/' + filename, 'w')
    f.write("P1\n")
    f.write("16 16\n")
    
    split_ex = [ex[i:i+16] for i in range(len(ex)-16) if i%16==0]
    
    for row in split_ex:
        for num in row:
            f.write(str(int(num)))
        f.write("\n")
    
    f.close()
    return f