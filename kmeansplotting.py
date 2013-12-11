import numpy as np
import pylab as pl

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from collections import defaultdict
from scipy.stats import mode

np.random.seed(42)

def clusterLabelDict(kmeans, data):
    cluster_label_dict = defaultdict(list)

    train = [row[:-1] for row in data.examples]
    scaled = scale(train)
    reduced_data = PCA(n_components=2).fit_transform(scaled)

    #  Get arbitrary cluster labels for each prediction.
    for i,row in enumerate(reduced_data):
        cluster = kmeans.predict(row)[0]            
        cluster_label_dict[cluster].append(data.examples[i][-1])
    
    #  Reset each key to the MODE of each inner list.
    for cluster in cluster_label_dict:
        m = mode(cluster_label_dict[cluster])[0][0]
        cluster_label_dict[cluster] = int(m)
        
    #  Return a normal dict, not a defaultdict<list>.
    return dict(cluster_label_dict)

###############################################################################
# Visualize the results on PCA-reduced data

def plot_pca(data):
    train = [row[:-1] for row in data.examples]
    scaled = scale(train)

    reduced_data = PCA(n_components=2).fit_transform(scaled)
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    kmeans.fit(reduced_data)

    cluster_label_dict = clusterLabelDict(kmeans, data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')

    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_

    for i in range(len(centroids)):
        c0 = centroids[:, 0][i]
        c1 = centroids[:, 1][i]
        predicted = kmeans.predict(centroids[i])
        label = cluster_label_dict[predicted[0]]
        pl.scatter(c0, c1, marker='$%d$' % label,
            s=169, linewidths=3, color='w', zorder=10)

    pl.title('K-means clustering on digits, reduced to 2-D with PCA\n'
         'Each white number is the mode of its centroid.')
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    pl.show()