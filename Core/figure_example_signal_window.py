import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imsave

import data

sample_rate = 1000
channels = range(0, 8)
#sequence_groups = data.words_10_20_sentences_dataset(include_surrounding=True, channels=channels)
sequence_groups = data.process(1, ['data/data/223_w10_s10.txt'], sample_rate=1000, channels=range(0, 8))

#sample_rate = 250
#channels = range(1, 8)
#labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]
#sequence_groups = data.process_scrambled(labels, ['eric2.txt'], channels=channels, sample_rate=250, surrounding=150)


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

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

#### Apply ricker wavelet subtraction
ricker_width = 35 * sample_rate // 250
ricker_sigma = 4.0 * sample_rate / 250
ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
ricker_subtraction_multiplier = 2.0
sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

low_freq = 0.5 #0.5
high_freq = 8 #8
order = 1

#### Apply soft bandpassing
sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

#### Apply hard bandpassing
#sequence_groups = data.transform.fft(sequence_groups)
#sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
#sequence_groups = np.real(data.transform.ifft(sequence_groups))

#sequence = sequence_groups[12][0]

#length = 3600
#print len(sequence_groups[12][0])
#sequence_groups = data.transform.pad_truncate(sequence_groups, length, position=0.56)
#sequence = sequence_groups[12][0]

#sequence = sequence_groups[0][1][2200:6500]
sequence = sequence_groups[0][1][2200:3450]


colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
#fig = plt.figure(figsize=(12, 3.5))
fig = plt.figure(figsize=(4, 2))
#plt.subplots_adjust(left=0.0525, right=0.995, bottom=0.13, top=0.93)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plt.title('Example Signal Window')
plt.xlabel('Sample')
plt.ylabel(u'Signal (\u03bcV)')
for i in range(len(channels)):
    plt.plot(sequence[:,i], c=colors[channels[i]], alpha=1.0, lw=1.5)
filename = 'figure_example_signal_window.png'
plt.savefig(filename, transparent=True)
plt.show()

image = imread(filename)
image = image[1:,1:]
imsave(filename, image)

