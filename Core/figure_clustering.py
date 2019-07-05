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


sequence_groups = data.digits_sequences_session_1_dataset(include_surrounding=True, channels=range(4, 8))
#sequence_groups = data.digits_dataset(include_surrounding=True)

#sequence_groups = data.digits_session_2_dataset(include_surrounding=True)
#sequence_groups = data.digits_and_silence_dataset(include_surrounding=False)

sample_rate = 250 #250


def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))
    
def transform_data(sequence_groups, sample_rate=250):
    #### Apply DC offset and drift correction
    drift_low_freq = 0.5 #0.5
    sequence_groups = data.transform.subtract_initial(sequence_groups)
    sequence_groups = data.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    sequence_groups = data.transform.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq
    notch_freq = 60
    num_times = 3 #pretty much just the filter order
    freqs = map(int, map(round, np.arange(1, sample_rate/(2. * notch_freq)) * notch_freq))
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = data.transform.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    #sequence_groups = data.transform.normalize_std(sequence_groups)

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
    ricker_subtraction_multiplier = 2.0
    sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

    #### Apply sine wavelet kernel
    #period = int(sample_rate)
    #sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
    #sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5 #0.5
    high_freq = 8 #8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)
    
    #### Apply hard bandpassing
    #sequence_groups = data.transform.fft(sequence_groups)
    #sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    #sequence_groups = np.real(data.transform.ifft(sequence_groups))
#    
    return sequence_groups

sequence_groups = transform_data(sequence_groups)

sequences, labels = data.get_inputs(sequence_groups)
num_channels = len(sequences[0][0])



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



#sequences = sequences[:50]
#labels = labels[:50]

print np.shape(sequences)
print np.shape(labels)


def window_split(sequence, window, stride=None, step=1):
    if stride is None: stride = window
    return np.array([np.concatenate(sequence[i:i+window:step]) for i in range(0, len(sequence)-window+1, stride)])
def center_window(sequence, window):
    start = (len(sequence)-window)//2
    return np.array(np.concatenate(sequence[start:start+window]))
def window_split_and_spans(sequence, window, stride=None, step=1):
    if stride is None: stride = window
    spans = map(lambda i: (i, i+window), range(0, len(sequence)-window+1, stride))
    return np.array([np.concatenate(sequence[i:j:step]) for i, j in spans]), spans


#num_components = 3
    
    
min_length = min(map(len, sequences))
print min_length

window_size = min(200, min_length) #250
print window_size

stride = 1 #5
fragment_span_pairs = map(lambda seq: window_split_and_spans(seq, window_size, stride, 1), sequences)
fragments, fragment_spans = np.transpose(fragment_span_pairs)
sequence_lengths = map(lambda spans: spans[-1][1] - spans[0][0], fragment_spans)
print sequence_lengths
fragment_labels = [[i] * len(fragments[i]) for i in range(len(fragments))]
fragments = np.concatenate(fragments)
centered_fragments = np.array(map(lambda fragment: fragment - np.mean(fragment), fragments))
fragment_spans = np.concatenate(fragment_spans)
fragment_labels = np.concatenate(fragment_labels)
    
#print np.shape(fragments)
#print np.shape(fragment_labels)
##model = PCA(n_components=num_components, verbose=1)
#model = TSNE(n_components=num_components, perplexity=30.0, n_iter=1000, verbose=2)
#reduced = model.fit_transform(fragments)

#reduced = fragments
#print np.shape(reduced)
#
#closest = np.zeros((len(reduced), len(reduced)))
#distances = np.zeros(np.shape(closest))
#ordered_distances = np.zeros(np.shape(closest))
#for i in range(len(reduced)):
#    print str(i+1) + '/' + str(len(reduced)), str((i+1)*100//len(reduced)) + '%'
#    ordered_dists = [np.linalg.norm(reduced[j] - reduced[i]) for j in range(len(reduced))]
#    ordered_distances[i] = ordered_dists
#    dists = sorted(zip(ordered_dists, range(len(reduced))))
#    distances[i], closest[i] = np.transpose(dists)
    
    
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
##z = linkage(ordered_distances, 'ward')
#z = linkage(fragments, 'ward')
#plt.figure(figsize=(16, 10))
#dendrogram(z, leaf_rotation=0.0, leaf_font_size=8.0)
##plt.show()

# Exploratory k-means clustering
#kmeans = KMeans(n_clusters=4).fit(ordered_distances)

np.random.seed(8)

num_clusters = 5 #5
kmeans = KMeans(n_clusters=num_clusters).fit(fragments)
cluster_indices = kmeans.labels_
centers = kmeans.cluster_centers_
centered_centers = np.array(map(lambda center: center - np.mean(center), centers))
cluster_counts = np.array(sorted(zip(*np.unique(cluster_indices, return_counts=True))))[:,1]
for i in range(num_clusters):
    print i, cluster_counts[i]
    

#fig, axes = plt.subplots(1, len(np.unique(cluster_indices)), figsize=(16, 3))
#plt.subplots_adjust(left=0.05, right=0.95)
#for i in np.unique(cluster_indices):
#    axes[i].plot(np.reshape(centers[i], (-1, num_channels)))
##    axes[i].plot(np.reshape(average_fragment, (-1, 1)))
##plt.show()


counts = [[np.array([0.0] * sequence_lengths[j]) for j in range(len(sequences))]\
          for i in range(num_clusters)]
for i in range(len(cluster_indices)):
    span_midpoint = (fragment_spans[i][0] + fragment_spans[i][1]) // 2
    start, end = fragment_spans[i][0], fragment_spans[i][1]
    start, end = span_midpoint - stride//2, span_midpoint + (stride - stride//2)
#    counts[cluster_indices[i]][fragment_labels[i]][start:end] += 1
    for k in range(num_clusters):
        counts[k][fragment_labels[i]][start:end] -= np.linalg.norm(fragments[i]-centers[k])
#        counts[k][fragment_labels[i]][start:end] -= np.linalg.norm(centered_fragments[i]-centered_centers[k])
    
cluster_probabilities = [np.array([[0.0] * num_clusters for j in range(sequence_lengths[i])])\
                         for i in range(len(sequences))]
for i in range(len(sequences)):
    for j in range(sequence_lengths[i]):
        total = sum([counts[k][i][j] for k in range(num_clusters)])
        for k in range(num_clusters):
            cluster_probabilities[i][j][k] = counts[k][i][j]# / total

example_index = 0 #0,3
#print [cluster_indices[i] for i in range(len(cluster_indices))\
#       if (i == 0 or cluster_indices[i] != cluster_indices[i-1]) and fragment_labels[i] == example_index]
sequence_clusters = [list(cluster_indices[np.where(fragment_labels == i)]) for i in range(len(sequences))]
print sequence_clusters[example_index]
for i in range(len(sequence_clusters)):
    filtered_clusters = []
    count = 0
    last = None
    for j in range(len(sequence_clusters[i])):
        if sequence_clusters[i][j] != last:
            last = sequence_clusters[i][j]
            count = 0
        count += 1
        if count >= 30 // stride and (not len(filtered_clusters) or \
                                      filtered_clusters[-1] != sequence_clusters[i][j]):
            filtered_clusters.append(sequence_clusters[i][j])
    sequence_clusters[i] = filtered_clusters
    
for i in range(len(sequence_clusters)):
    print sequence_clusters[i]
    
print sequence_clusters[example_index]
            
#print np.argmax(cluster_probabilities[example_index], axis=1)
#fig = plt.subplots(2, figsize=(16, 9))
fig = plt.subplots(2, figsize=(12, 7))
#plt.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.97, wspace=0.20, hspace=0.35)
plt.subplots_adjust(left=0.06, right=0.99, bottom=0.0725, top=0.955, wspace=0.20, hspace=0.50)
cluster_min, cluster_max = 1.1 * np.min(centers), 1.1 * np.max(centers)
plt.subplot(3, 1, 1)
for i in range(num_channels):
#    plt.plot(sequences[example_index][:sequence_lengths[example_index],i], label='Channel ' + str(i+1))
    plt.plot(sequences[example_index][window_size//2:sequence_lengths[example_index]-window_size//2,i], label='Channel ' + str(i+1))
plt.title('Example Sequence')
plt.xlabel('Sample index')
#plt.ylabel('Sequence value')
plt.subplot(3, 1, 2)
for i in range(num_clusters):
#    plt.plot(cluster_probabilities[example_index][:,i], label='Cluster ' + str(i))
    plt.plot(cluster_probabilities[example_index][window_size//2:-window_size//2,i], label='Cluster ' + str(i))
plt.title('Cluster Representation (num_clusters='+str(num_clusters)+', window_size='+str(window_size)+')')
plt.xlabel('Window center')
plt.ylabel('Neg. dist. between cluster and window')
plt.legend(loc=3)
for i in range(num_clusters):
    plt.subplot(3, num_clusters, 2*num_clusters + i+1)
    plt.plot(np.reshape(centers[i], (-1, num_channels)))
    plt.ylim((cluster_min, cluster_max))
    plt.title('Cluster ' + str(i) + ' (n='+str(cluster_counts[i])+')')
    print cluster_counts[i]
    plt.xlabel('Window sample index')

min_length = 2
max_length = 6
group_counts = {}
group_count_pairs = []
for length in range(min_length-1, max_length):
    for sequence in sequence_clusters:
        for i in range(len(sequence) - length):
            subsequence = tuple(sequence[i:i+length+1])
            group_counts[subsequence] = group_counts.get(subsequence, 0) + 1
for key in group_counts:
    group_count_pairs.append((group_counts[key], key))
group_count_pairs = sorted(group_count_pairs)[::-1]
print
for pair in group_count_pairs:
    print pair
    
#plt.savefig('figure_clustering.png')
    

# TODO NEXT (instead of using convolutions, program a per-channel distance metric)
diff_kernel = np.array([-1, 1])
def align(sequence1, sequence2, tail_dist=200):
#    return np.transpose(np.array([data.transform.convolve(sequence1[:,i], sequence2[:,i])\
#                                  for i in range(num_channels)]))
    diffs = []
    seq2 = np.concatenate([sequence2, data.transform.correlate(sequence2, diff_kernel)], axis=1)
#    plt.figure()
#    plt.plot(data.transform.correlate(sequence1, diff_kernel)[1:], lw=0.5)
#    plt.plot(data.transform.correlate(sequence2, diff_kernel)[1:], lw=0.5)
#    plt.title('TEST')
#    plt.show()
    for i in range(len(sequence1)-len(sequence2)):
        seq1 = sequence1[i:i+len(sequence2)]
        seq1 = np.concatenate([seq1, data.transform.correlate(seq1, diff_kernel)], axis=1)
        diffs.append(np.linalg.norm(seq1[1:] - seq2[1:], axis=0))
    diffs = list(np.array(diffs) / np.std(diffs, axis=0) * [1., 1., 1., 1., 0., 0., 0., 0.]);  # INVESTIGATE
    diffs = [np.array([float('inf')] * (num_channels*2)) for _ in range(len(sequence2)//2)] + diffs +\
            [np.array([float('inf')] * (num_channels*2)) for _ in range(len(sequence2)-len(sequence2)//2)]
    return np.argmin(np.sum(diffs, axis=1)[-tail_dist:]) + len(diffs) - tail_dist, diffs
def merge(sequence1, sequence2, index):
    tmp1 = int(min(len(sequence1),index+len(sequence2)))
    tmp2 = int(min(len(sequence2),len(sequence1)-index))
    merged = np.concatenate([sequence1[:index,:], (sequence1[index:tmp1,:] + sequence2[:tmp2,:])/2.0, sequence2[tmp2:,:]], axis=0)
    return merged

#print len(group_count_pairs)
#for id, (count, test_sequence) in enumerate(group_count_pairs):
##    print id, count, test_sequence
#
#    current = np.reshape(centers[test_sequence[0]], (-1, num_channels))
#    for i in range(1, len(test_sequence)):
#        index, aligned = align(current,
#                               np.reshape(centers[test_sequence[i]][:3*num_channels], (-1, num_channels)))
#        #plt.figure()
#        #plt.plot(np.reshape(centers[test_sequence[0]], (-1, num_channels)))
#        #plt.figure()
#        #plt.plot(np.reshape(centers[test_sequence[1]], (-1, num_channels)))
##        print index
##        plt.figure()
##        plt.plot(aligned)
#        reshaped2 = np.reshape(centers[test_sequence[i]], (-1, num_channels))
#    #    plt.plot(np.arange(len(reshaped1)), reshaped1[:,i])
#    #    plt.plot(np.arange(len(reshaped1)), reshaped1[:,i])
#        current = merge(current, reshaped2, index)
#    tmp = np.abs(data.transform.correlate(current, diff_kernel))
#    tmp2 = [[0.0] * num_channels]
#    for i in range(1, len(tmp)):
#        tmp2.append([min(tmp[i-1,c], tmp[i,c]) for c in range(num_channels)])
#    tmp2 = np.array(tmp2)
#    maxes = np.max(np.concatenate([tmp, data.transform.correlate(tmp2, diff_kernel)], axis=1)[1:], axis=0)
##    print maxes[:num_channels]
##    print maxes[num_channels:num_channels*2]
#    max1 = max(maxes[:num_channels])
#    max2 = max(maxes[num_channels:num_channels*2])
##    print max1, max2
#    if max1 < 1.0 and max2 < 0.15: #1.0, 0.15
#        print id, test_sequence
#        print max1, max2
#        plt.figure()
#        for i in range(num_channels):
#            plt.plot(current[:,i], label='Channel ' + str(i+1))
#        plt.legend()
#        plt.title('Feature from combined clusters ' + str(test_sequence))
#        plt.xlabel('Feature index')
#        plt.ylabel('Feature value')
    
plt.show()



