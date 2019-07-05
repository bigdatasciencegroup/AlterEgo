import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import math

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

#sequence_groups = data.transform.normalize_std(sequence_groups)

#low = 0.5 #0.5
#high = 8 #8
#order = 1 #1
#sequence_groups = data.transform.bandpass_filter(sequence_groups, low, high, sample_rate, order=order)

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
sin_kernel = normalize_kernel(np.sin(np.arange(250)/250. * 1*np.pi), subtract_mean=True)
sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)


sequences, labels = data.get_inputs(sequence_groups)

#sequences = sequences[:50]
#labels = labels[:50]

print np.shape(sequences)
print np.shape(labels)


def window_split(sequence, window, stride=None, step=1):
    if stride is None: stride = window
    return np.array([np.concatenate(sequence[i:i+window:step,:]) for i in range(0, len(sequence)-window+1, stride)])
def center_window(sequence, window):
    start = (len(sequence)-window)//2
    return np.array(np.concatenate(sequence[start:start+window]))


num_components = 3
    
    
fig = None
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.97)
min_length = min(map(len, sequences))
print min_length
for i in range(249, min_length):
#    fragments = map(lambda seq: window_split(seq, i+1, i+1, 1), sequences)
#    fragment_labels = [[labels[i]] * len(fragments[i]) for i in range(len(fragments))]
#    fragments = np.concatenate(fragments)
#    fragment_labels = np.concatenate(fragment_labels)

    fragments = map(lambda seq: center_window(seq, i+1), sequences)
    fragment_labels = labels
    
    print np.shape(fragments)
    print np.shape(fragment_labels)
#    model = PCA(n_components=num_components)
    model = TSNE(n_components=num_components, perplexity=30.0, n_iter=3000, verbose=2)
    reduced = model.fit_transform(fragments)
    print np.shape(reduced)
    print np.transpose(reduced[np.where(fragment_labels == 0)])

    
    closest = np.zeros((len(reduced), len(reduced) - 1))
    for i in range(len(reduced)):
        dists = sorted([(np.linalg.norm(reduced[j] - reduced[i]), j) for j in range(len(reduced)) if i != j])
        closest[i] = np.array(dists)[:, 1]
    closest_labels = np.array(map(lambda others: map(lambda k: fragment_labels[k], others), closest))
#    print closest_labels
#    for i in range(len(closest_labels)):
#        print fragment_labels[i], closest_labels[i][:20]
    reduced, fragment_labels = zip(*np.array([(reduced[i], fragment_labels[i]) for i in range(len(reduced))\
                                              if np.all(closest_labels[i][:0] == fragment_labels[i])]))
    reduced = np.array(reduced)
    fragment_labels = np.array(fragment_labels)
    print np.shape(reduced)
    print np.shape(fragment_labels)
        
    


    if num_components == 2:
        if fig is None:
            fig = plt.figure(figsize=(16, 9))
        plt.clf()
        plt.title(str(i+1))
        print i+1
        for i in sorted(np.unique(labels)):
            plt.scatter(*np.transpose(reduced[np.where(fragment_labels == i)]),
    #                    marker=',', s=(72./plt.gcf().dpi)**2, alpha=1.0,
                        marker='.', alpha=0.5, s=200,
                        label=str(i))
        plt.legend()

    #    heatmap, xedges, yedges = np.histogram2d(*np.transpose(reduced), bins=200)
    ##    heatmap = np.log(heatmap)
    ##    heatmap[np.where(heatmap == -np.inf)] = 0.0
    #    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #    plt.imshow(heatmap.T, extent=extent, origin='lower')


    #    plt.savefig('pca_output/' + str(i+1) + '.jpg')
    #    plt.pause(0.000001)
        plt.show()
        kill
    
    elif num_components == 3:
        traces = []
        for i in sorted(np.unique(labels)):
            x, y, z = np.transpose(reduced[np.where(fragment_labels == i)])
            traces.append(go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       marker=dict(size=5, opacity=0.5,
                                                   line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5))))
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=traces, layout=layout)
        plotly.offline.plot(fig)
        kill

