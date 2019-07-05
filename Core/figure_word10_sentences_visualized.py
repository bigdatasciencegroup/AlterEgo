import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))
    
def transform_data(sequence_groups, sample_rate=1000):
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
    
sequence_groups = transform_data(data.words_10_20_sentences_dataset(include_surrounding=True, surrounding=150))

sequence_groups = data.transform.downsample(sequence_groups, 4)

#length = 10000
#sequence_groups = data.transform.pad_truncate(sequence_groups, length)

lens = map(len, data.get_inputs(sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

words = np.array(['i', 'am', 'cold', 'hot', 'hungry', 'tired', 'want', 'need', 'food', 'water'])
label_map = [
    [2, 9, 6, 4, 0],
    [9, 5, 3, 0, 8],
    [8, 4, 7, 3, 5],
    [2, 3, 6, 9, 1],
    [7, 6, 5, 2, 1],
    [3, 8, 5, 1, 4],
    [4, 9, 7, 0, 6],
    [9, 4, 5, 7, 8],
    [8, 3, 1, 2, 0],
    [9, 0, 1, 7, 5],
    [6, 3, 2, 4, 1],
    [4, 6, 1, 9, 2],
    [0, 7, 2, 5, 8],
    [6, 8, 0, 3, 9],
    [8, 2, 7, 1, 0],
    [5, 4, 8, 1, 3],
    [7, 4, 2, 6, 0],
    [2, 8, 6, 7, 9],
    [1, 5, 9, 3, 4],
    [5, 0, 4, 3, 7],
]

fig, axes = plt.subplots(10, 2, figsize=(8, 10))
#fig.subplots_adjust(left=0.08, bottom=0.03, right=0.97, top=0.92, wspace=0.2, hspace=1.2)
fig.subplots_adjust(left=0.08, bottom=0.01, right=0.97, top=0.92, wspace=0.2, hspace=0.75)
plt.suptitle('Example Recordings from the 10-Word Vocab, 20-Sentence Dataset', fontweight='bold')
colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
for i in range(len(sequence_groups)):
    x, y = divmod(i, 2)
    axes[x][y].set_title('"' + ' '.join(map(lambda x: words[x], label_map[i])) + '"')
    seq = sequence_groups[i][0]
#    seq = np.mean(sequence_groups[i], axis=0)
    for c in range(8):
        axes[x][y].plot(seq[:,c], c=colors[c], lw=0.5)
    axes[x][y].set_ylim((-70, 70))
    axes[x][y].set_ylabel(u'\u03BCV', rotation=0)
    axes[x][y].get_xaxis().set_ticks([])
    axes[x][y].tick_params(axis='both', which='major', labelsize=8)

plt.savefig('figure_word10_sentences_visualized.png')

plt.show()