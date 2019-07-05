import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import math
import fnmatch
import os

import data


sequence_groups = data.digits_dataset(include_surrounding=False)
#sequence_groups = data.digits_session_2_dataset(include_surrounding=True)
#sequence_groups = data.digits_and_silence_dataset(include_surrounding=False)

sample_rate = 250 #250
drift_low_frequency = 0.5 #0.5
sequence_groups = data.transform.subtract_initial(sequence_groups)
sequence_groups = data.transform.highpass_filter(sequence_groups, drift_low_frequency, sample_rate)
sequence_groups = data.transform.subtract_mean(sequence_groups)

notch1 = 59.9 #59.9
notch2 = 119.8 #119.8
sequence_groups = data.transform.notch_filter(sequence_groups, notch2, sample_rate)
sequence_groups = data.transform.notch_filter(sequence_groups, notch1, sample_rate)
sequence_groups = data.transform.notch_filter(sequence_groups, notch2, sample_rate)
sequence_groups = data.transform.notch_filter(sequence_groups, notch1, sample_rate)

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

ricker_kernel = normalize_kernel(ricker_wavelet(35, 4.0))
ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
ricker_subtraction_multiplier = 2
sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

#sequence_groups = data.transform.normalize_std(sequence_groups)

low = 0.5 #0.5
high = 8 #8
order = 1 #1

sequence_groups = data.transform.bandpass_filter(sequence_groups, low, high, sample_rate, order=order)

#sin_kernel = normalize_kernel(np.sin(np.arange(250)/250. * 1*np.pi), subtract_mean=True)
#sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)


sequences, labels = data.get_inputs(sequence_groups)



# Exploratory (TIMIT)
#def load_data(data_dir, labels_dir):
#    print 'Loading data from', data_dir
#    matches = []
#    for root, dirnames, filenames in os.walk(data_dir):
#        for filename in fnmatch.filter(filenames, '*.npy'):
#            matches.append(os.path.join(root, filename))
#    sequences = np.array(map(lambda filepath: list(np.transpose(np.load(filepath))), sorted(matches)))
#    print 'Loading labels from', labels_dir
#    matches = []
#    for root, dirnames, filenames in os.walk(labels_dir):
#        for filename in fnmatch.filter(filenames, '*.npy'):
#            matches.append(os.path.join(root, filename))
##    print matches
#    labels = np.array(map(lambda filepath: list(np.load(filepath)), sorted(matches)))
#    return sequences, labels
#sequences, labels = load_data('timit/tensorflow_CTC_example/train_data/data',
#                              'timit/tensorflow_CTC_example/train_data/labels')



# Each sequence has its own label, to prevent data overlap matching
labels = np.arange(len(sequences))

sequences = sequences[:50]
labels = labels[:50]

print np.shape(sequences)
print np.shape(labels)
print map(len, sequences)


def window_split(sequence, window, stride=None, step=1):
    if stride is None: stride = window
    return np.array([np.concatenate(sequence[i:i+window:step]) for i in range(0, len(sequence)-window+1, stride)])
def center_window(sequence, window):
    start = (len(sequence)-window)//2
    return np.array(np.concatenate(sequence[start:start+window]))


#num_components = 3
    
    
min_length = min(map(len, sequences))
print min_length

window_size = min(250, min_length)
print window_size

#fragments = map(lambda seq: window_split(seq, window_size, window_size//4, 1), sequences)
fragments = map(lambda seq: window_split(seq, window_size, 5, 1), sequences)
fragment_labels = [[labels[i]] * len(fragments[i]) for i in range(len(fragments))]
fragments = np.concatenate(fragments)
fragment_labels = np.concatenate(fragment_labels)

#fragments = map(lambda seq: center_window(seq, window_size), sequences)
#fragment_labels = labels
    
#print np.shape(fragments)
#print np.shape(fragment_labels)
##model = PCA(n_components=num_components, verbose=1)
#model = TSNE(n_components=num_components, perplexity=30.0, n_iter=1000, verbose=2)
#reduced = model.fit_transform(fragments)

reduced = fragments
print np.shape(reduced)

closest = np.zeros((len(reduced), len(reduced)))
distances = np.zeros(np.shape(closest))
ordered_distances = np.zeros(np.shape(closest))
for i in range(len(reduced)):
    print str(i+1) + '/' + str(len(reduced)), str((i+1)*100//len(reduced)) + '%'
    ordered_dists = [np.linalg.norm(reduced[j] - reduced[i]) for j in range(len(reduced))]
    ordered_distances[i] = ordered_dists
    dists = sorted(zip(ordered_dists, range(len(reduced))))
    distances[i], closest[i] = np.transpose(dists)
    
    
# Exploratory PCA/t-SNE
#fragments = ordered_distances
##model = PCA(n_components=3)
#model = TSNE(n_components=3, perplexity=30.0, n_iter=1000, verbose=2)
#reduced = model.fit_transform(fragments)
#
#traces = []
#x, y, z = np.transpose(reduced)
#traces = [go.Scatter3d(x=x, y=y, z=z,
#                       mode='markers',
#                       marker=dict(size=5, opacity=0.5,
#                                   line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5)))]
#layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
#fig = go.Figure(data=traces, layout=layout)
#plotly.offline.plot(fig)
#kill

# Exploratory hierarchical clustering
#z = linkage(ordered_distances, 'ward')
z = linkage(fragments, 'ward')
plt.figure(figsize=(16, 10))
dendrogram(z, leaf_rotation=0.0, leaf_font_size=8.0)
#plt.show()

# Exploratory k-means clustering
#kmeans = KMeans(n_clusters=4).fit(ordered_distances)
kmeans = KMeans(n_clusters=6).fit(fragments)
cluster_indices = kmeans.labels_
for x in np.unique(cluster_indices):
    print x, len(filter(lambda ind: ind == x, cluster_indices))
print kmeans.cluster_centers_

fig, axes = plt.subplots(1, len(np.unique(cluster_indices)), figsize=(16, 3))
plt.subplots_adjust(left=0.05, right=0.95)
for i in np.unique(cluster_indices):
#    average_fragment = np.mean(fragments[np.where(cluster_indices == i)], axis=0)
    average_fragment = kmeans.cluster_centers_[i]
    axes[i].plot(np.reshape(average_fragment, (-1, 4)))
#    axes[i].plot(np.reshape(average_fragment, (-1, 1)))
plt.show()

#kill


average_distances = map(lambda dists: np.mean(dists[np.where(dists >= 0)][:20]), distances)
average_distance_pairs = list(sorted(zip(average_distances, range(len(average_distances)))))
for pair in average_distance_pairs[:10]:
    print pair

num_rows = 6
num_cols = 10
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 9))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.40)
displayed_label_set = set()
displayed_index = 0
for i in range(num_rows):
    while fragment_labels[average_distance_pairs[displayed_index][1]] in displayed_label_set:
        displayed_index += 1
    if displayed_index >= len(average_distance_pairs): break
    print displayed_index
    test_index = average_distance_pairs[displayed_index][1]
    displayed_label_set.add(fragment_labels[int(test_index)])
    sorted_indices = closest[test_index][np.where(distances[test_index] >= 0)]
    axes[i][0].plot(np.reshape(reduced[test_index], (-1, 4)))
    axes[i][0].set_title(fragment_labels[int(test_index)])
    other_index = 0
    label_set = set([fragment_labels[int(test_index)]])
    for j in range(20):
        if other_index >= len(sorted_indices): break
        while fragment_labels[int(sorted_indices[other_index])] in label_set:
            other_index += 1
        label = fragment_labels[int(sorted_indices[other_index])]
        label_set.add(label)
        if j < num_cols-1:
            displayed_label_set.add(label)
            axes[i][j+1].plot(np.reshape(reduced[int(sorted_indices[other_index])], (-1, 4)))
            axes[i][j+1].set_title(label)
        other_index += 1
plt.show()
    
#closest_labels = np.array(map(lambda others: map(lambda k: fragment_labels[k], others), closest))
##print closest_labels
##for i in range(len(closest_labels)):
##    print fragment_labels[i], closest_labels[i][:20]
#reduced, fragment_labels = zip(*np.array([(reduced[i], fragment_labels[i]) for i in range(len(reduced))\
#                                          if np.all(closest_labels[i][:0] == fragment_labels[i])]))
#reduced = np.array(reduced)
#fragment_labels = np.array(fragment_labels)
#print np.shape(reduced)
#print np.shape(fragment_labels)
