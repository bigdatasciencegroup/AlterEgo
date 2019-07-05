import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imsave

import data

sample_rate = 1000
channels = range(0, 8)
#channels = [1, 3, 6]

sequence_groups = data.words_10_20_sentences_dataset(include_surrounding=True, channels=channels)


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

sequence = sequence_groups[12][0]
colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
fig, axes = plt.subplots(len(channels), 1, figsize=(14, min(2+1.5*len(channels), 10)))
plt.subplots_adjust(hspace=0.0225)
for i in range(len(channels)):
    axes[i].plot(sequence[:,i], c=colors[channels[i]])
#regions = [(600, 1850), (1850, 3150), (3150, 4700), (4700, 6000), (6000, 7700)]
regions = [(600, 1950), (1950, 3150), (3150, 4700), (4700, 6000), (6000, 7700)]
#region_colors = ['red', 'orange', 'yellow', 'green', 'blue']
region_colors = ['#bbbbff', '#bbffbb', '#bbbbff', '#bbffbb', '#bbbbff']
for i in range(len(channels)):
    xlim, ylim = axes[i].get_xlim(), axes[i].get_ylim()
    for j in range(len(regions)):
        axes[i].fill_between(regions[j], [-1e9, -1e9], [1e9, 1e9],
                             facecolor=region_colors[j], alpha=0.15, interpolate=True)
        axes[i].plot([regions[j][0], regions[j][0]], [-1e9, 1e9], lw=2, c='black')
    axes[i].plot([regions[-1][1], regions[-1][1]], [-1e9, 1e9], lw=2, c='black')
    axes[i].set_xlim(xlim)
    axes[i].set_ylim(ylim)
    
fig = plt.figure(figsize=(14, 2))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.95)
#xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
for j in range(len(regions)):
    plt.fill_between(regions[j], [-1e9, -1e9], [1e9, 1e9],
                           facecolor=region_colors[j], alpha=0.15, interpolate=True)
#for i in [6]:
for i in range(8):
    plt.plot(sequence[:,i], c=colors[channels[i]])
for j in range(len(regions)):
    plt.plot([regions[j][0], regions[j][0]], [-1e9, 1e9], lw=2, c='black')
plt.plot([regions[-1][1], regions[-1][1]], [-1e9, 1e9], lw=2, c='black')
plt.gca().set_ylim((-60, 60))
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
filename = 'figure_example_signal_alignment.png'
plt.savefig(filename, transparent=True)

plt.show()

image = imread(filename)
image = image[1:,1:]
imsave(filename, image)






#sequences = []
#sequences.append(sequence_groups[12][0])
#sequences.append(sequence_groups[12][1])
#sequences.append(sequence_groups[12][2])
#sequences.append(sequence_groups[12][3])
#sequences.append(sequence_groups[12][4])
#sequences.append(sequence_groups[12][5])
#sequences.append(sequence_groups[12][6])
#sequences.append(sequence_groups[12][7])

#window_size = int(round(9.0 * sample_rate / 600))
#double_averaged = data.transform.window_average(sequence, window_size)
#double_averaged = data.transform.window_average(double_averaged, window_size)
#sequences.append(double_averaged)
#
#high_frequency = sequence - double_averaged
#sequences.append(high_frequency)
#
#rectified = np.abs(high_frequency)
#sequences.append(rectified)
#
#print np.mean(sequence, axis=0)
#print np.mean(np.square(double_averaged), axis=0)
#print np.mean(np.square(rectified), axis=0)
#
#zero_cross_counts = np.array([0] * len(channels))
#for c in range(len(channels)):
#    for i in range(len(high_frequency)-1):
#        if np.sign(high_frequency[i-1,c]) == -np.sign(high_frequency[i,c]):
#            zero_cross_counts[c] += 1
#print zero_cross_counts
#
#
#fft_width = sample_rate // 16 #16
#fourier_spectrums = [data.transform.fft(sequence[i:i+fft_width]) for i in range(0, len(sequence)-fft_width+1,32)]#32
#fourier_spectrums = np.abs(fourier_spectrums)[:,:fft_width//2,:]
#diff_fourier_spectrums = [fourier_spectrums[i+1] - fourier_spectrums[i] for i in range(len(fourier_spectrums)-1)]
#fourier_spectrums = np.transpose(fourier_spectrums, axes=(1, 0, 2))
#diff_fourier_spectrums = np.transpose(diff_fourier_spectrums, axes=(1, 0, 2))
#
#
#fig, axes = plt.subplots(len(channels), 2, figsize=(16, min(2+1.5*len(channels), 10)))
#plt.subplots_adjust(left=0.04, right=0.96, bottom=0.02, top=0.98, wspace=0.08, hspace=0.04)
#for c in range(len(channels)):
#    axes[c][0].imshow(np.log(fourier_spectrums[:,:,c]), cmap='magma')
#    axes[c][1].imshow(diff_fourier_spectrums[:,:,c], cmap='magma')

#colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
#
#fig, axes = plt.subplots(len(sequences), 1, figsize=(14, min(2+1.5*len(sequences), 10)))
#for i in range(len(sequences)):
#    for j in range(len(channels)):
#        axes[i].plot(sequences[i][:,j], c=colors[channels[j]])
#plt.show()

