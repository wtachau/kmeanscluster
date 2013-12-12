import numpy as np
import matplotlib.pyplot as plt

from learning import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from collections import defaultdict
from scipy.stats import mode
from scipy.spatial.distance import euclidean

import random
from pprint import pprint

import pdb

from kmeansplotting import plot_pca

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
        #print ex
        scaled = scale(ex[:-1])
        cluster = kmeans.predict(scaled)[0]   
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


# For each cluster, generate random subset of both correctly and 
# incorrectly classified data
def generateRandomSubset():
    data = DataSet(name="semeion")
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

    classifications = []

    for cc in kmeans.cluster_centers_:
        predicted = kmeans.predict(cc)
        label = cluster_label_dict[predicted[0]]

        ##  Find closest ex's to each centroid.
        distance = float('inf')
        closest = None
        closest_index = 0

        for i, ex in enumerate(scaled):
            d = euclidean(ex, cc)
            if d < distance:
                closest = ex
                closest_index = i
                distance = d

    # for i in range(10):
        correct_class = []
        incorrect_class = []
        for ind, ex in enumerate(data.examples):
            pred = predict(ex)
            actual = ex[-1]
            if (pred == label):
                if (pred == actual):
                    correct_class.append(ind)
                else:
                    incorrect_class.append(ind)
        classifications.append({'num':label, 'correct':correct_class,
            'incorrect':incorrect_class, 'closest':closest_index})
    
    # now generate random subset
    for num in classifications:
        num_subset = 5
        # for correct
        if (len(num['correct']) >= 5):
            num['correct_subset'] = random.sample(num['correct'], num_subset)
        else:
            num['correct_subset'] = num['correct']
        # and for incorrect
        if (len(num['incorrect']) >= 5):
            num['incorrect_subset'] = random.sample(num['incorrect'], num_subset)
        else:
            num['incorrect_subset'] = num['incorrect']

    # at this point classifications has everything -> can print or write to file
    
    print "Classificatons: "
    pprint(classifications)

    ##  Write PBM files.
    ##  For each centroid, save closest ex, 5 correct, and 5 incorrect ex's.

    for num in classifications:
        print num['num']
        print num['correct_subset']
        print num['incorrect_subset']
        
        for ex in num['correct_subset']:
            filename = "num"+str(num['num'])+"ex"+str(ex)+"correct"
            # pdb.set_trace()
            writePBM(data.examples[ex], filename=filename)

        for ex in num['incorrect_subset']:
            filename = "num"+str(num['num'])+"ex"+str(ex)+"incorrect"
            writePBM(data.examples[ex], filename=filename)

        closest_ex = num['closest']
        print closest_ex
        # closest_ex = closest_ex[0]
        filename = "num"+str(num['num'])+"ex"+str(closest_ex)+"closest"
        writePBM(data.examples[closest_ex], filename=filename)




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

def plot_clusters():
    data = semeionData()
    plot_pca(data)


def writePBM(ex, filename='example'):
    """Takes a binary feature vector and writes a PBM file."""
    f = file('/Users/SDA/Fall 2013/CS158/kmeanscluster/' + filename + '.pbm', 'w')
    f.write("P1\n")
    f.write("16 16\n")
    
    split_ex = [ex[i:i+16] for i in range(len(ex)-16) if i%16==0]
    
    for row in split_ex:
        for num in row:
            f.write(str(int(num)))
        f.write("\n")
    
    f.close()
    return f




generateRandomSubset()

