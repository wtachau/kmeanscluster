import numpy as np
import matplotlib.pyplot as plt

from learning import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from collections import defaultdict
from scipy.stats import mode
from scipy.spatial.distance import euclidean

import pdb


def semeionData():
    data = DataSet(name="semeion")
    return data


def kmeansLearner(data):
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    train = [row[:-1] for row in data.examples]
    scaled = scale(train)
    kmeans.fit(scaled)

    cluster_label_dict = clusterLabelDict(kmeans, data)

    def predict(ex):
        cluster = kmeans.predict(ex[:-1])[0]   
        return cluster_label_dict[cluster]
        
    #  Pass `predict` function to cross_validation later.
    return predict

def showRepDigits(data):
    # first train on all the data
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    train = [row[:-1] for row in data.examples]
    scaled = scale(train)
    kmeans.fit(scaled)
    
    cluster_label_dict = clusterLabelDict(kmeans, data)

    def predict(ex):
        scaled = scale(ex[:-1])
        cluster = kmeans.predict(scaled)[0]   
        return cluster_label_dict[cluster]

    cluster_indices = []

    for cluster_center in kmeans.cluster_centers_:
        print cluster_center

        distance = float('inf')
        closest = None
        closest_index = 0

        for i, ex in enumerate(scaled):
            d = euclidean(ex, cluster_center)
            if d < distance:
                closest = ex
                closest_index = i
                distance = d

            # closest = min(closest, euclidean(ex, cluster_center))

        print "CLOSEST INDEX @ >>" + str(closest_index)
        cluster_indices.append({'ind':closest_index, 'pred':predict(data.examples[closest_index])})

    print cluster_indices

    #now write each to file
    


def clusterLabelDict(kmeans, data):
    cluster_label_dict = defaultdict(list)
    
    #  Get arbitrary cluster labels for each prediction.
    for row in data.examples:
        cluster = kmeans.predict(row[:-1])[0]            
        cluster_label_dict[cluster].append(row[-1])
    
    #  Reset each key to the MODE of each inner list.
    for cluster in cluster_label_dict:
        m = mode(cluster_label_dict[cluster])[0][0]
        cluster_label_dict[cluster] = int(m)
        
    #  Return a normal dict, not a defaultdict<list>.
    return dict(cluster_label_dict)


if __name__ == '__main__':

    data = semeionData()
    #results = cross_validation(kmeansLearner, data)
    #print "Kmeans accuracy: ", results

    showRepDigits(data)



