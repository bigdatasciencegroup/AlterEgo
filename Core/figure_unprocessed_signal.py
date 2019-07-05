import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

#sample_rate = 1000
#channels = range(0, 8)
#sequence_groups = data.words_10_20_sentences_dataset(include_surrounding=True, channels=channels)

sample_rate = 250
channels = range(1, 8)
labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]
sequence_groups = data.process_scrambled(labels, ['eric2.txt'], channels=channels, sample_rate=250, surrounding=150)


#### Apply DC offset and drift correction
drift_low_freq = 0.5 #0.5
sequence_groups = data.transform.subtract_initial(sequence_groups)
#sequence_groups = data.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
#sequence_groups = data.transform.subtract_mean(sequence_groups)

#### Apply notch filters at multiples of notch_freq
#notch_freq = 60
#num_times = 3 #pretty much just the filter order
#freqs = map(int, map(round, np.arange(1, sample_rate/(2. * notch_freq)) * notch_freq))
#for _ in range(num_times):
#    for f in reversed(freqs):
#        sequence_groups = data.transform.notch_filter(sequence_groups, f, sample_rate)

#### Apply standard deviation normalization
#sequence_groups = data.transform.normalize_std(sequence_groups)

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

#### Apply ricker wavelet subtraction
#ricker_width = 35 * sample_rate // 250
#ricker_sigma = 4.0 * sample_rate / 250
#ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
#ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
#ricker_subtraction_multiplier = 2.0
#sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

low_freq = 0.5 #0.5
high_freq = 8 #8
order = 1

#### Apply soft bandpassing
#sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

#### Apply hard bandpassing
#sequence_groups = data.transform.fft(sequence_groups)
#sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
#sequence_groups = np.real(data.transform.ifft(sequence_groups))

#sequence = sequence_groups[12][0]

#length = 450
#sequence_groups = data.transform.pad_truncate(sequence_groups, length)
sequence = sequence_groups[0][0] #0,6,9
#avg = np.mean(sequence_groups[0], axis=0)
#distances = [np.linalg.norm(sequence - avg) for sequence in sequence_groups[0]]
#print np.argmin(distances)
#print np.argmax(distances)


colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
fig = plt.figure(figsize=(12, 3.5))
plt.subplots_adjust(left=0.05, right=0.995, bottom=0.13, top=0.93)
plt.title('Unprocessed Signal ("zero", 7 channels)')
plt.xlabel('Sample')
plt.ylabel(u'Signal (\u03bcV)')
for i in range(len(channels)):
    plt.plot(sequence[:,i], c=colors[channels[i]], alpha=1.0, lw=1.0)
plt.savefig('figure_unprocessed_signal.png')
plt.show()

#colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
#fig, axes = plt.subplots(len(channels), 1, figsize=(14, min(2+1.5*len(channels), 10)))
#plt.subplots_adjust(hspace=0.0225)
#for i in range(len(channels)):
#    axes[i].plot(sequence[:,i], c=colors[channels[i]])
#regions = [(600, 1850), (1850, 3150), (3150, 4700), (4700, 6000), (6000, 7700)]
#region_colors = ['red', 'orange', 'yellow', 'green', 'blue']
#for i in range(len(channels)):
#    xlim, ylim = axes[i].get_xlim(), axes[i].get_ylim()
#    for j in range(len(regions)):
#        axes[i].fill_between(regions[j], [-1e9, -1e9], [1e9, 1e9],
#                             facecolor=region_colors[j], alpha=0.15, interpolate=True)
#        axes[i].plot([regions[j][0], regions[j][0]], [-1e9, 1e9], lw=2, c='black')
#    axes[i].plot([regions[-1][1], regions[-1][1]], [-1e9, 1e9], lw=2, c='black')
#    axes[i].set_xlim(xlim)
#    axes[i].set_ylim(ylim)
#    
#fig = plt.figure(figsize=(14, 2))
#plt.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.95)
##xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
#for j in range(len(regions)):
#    plt.fill_between(regions[j], [-1e9, -1e9], [1e9, 1e9],
#                           facecolor=region_colors[j], alpha=0.15, interpolate=True)
#for i in [6]:
#    plt.plot(sequence[:,i], c=colors[channels[i]])
#for j in range(len(regions)):
#    plt.plot([regions[j][0], regions[j][0]], [-1e9, 1e9], lw=2, c='black')
#plt.plot([regions[-1][1], regions[-1][1]], [-1e9, 1e9], lw=2, c='black')
#plt.gca().set_ylim((-60, 60))
#plt.show()

