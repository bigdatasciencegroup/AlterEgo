import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

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
    
    return sequence_groups


#channels = range(1, 8)
channels = range(4, 8)
channels = [4]
#channels = [2, 4, 5, 6, 7]
surrounding = 250

labels = [14, 12, 5, 3, 1, 9, 2, 12, 2, 6, 0, 13, 12, 2, 13, 12, 9, 9, 10, 7, 12, 8, 7, 9, 1, 0, 14, 3, 9, 14, 11, 0, 7, 9, 5, 5, 4, 5, 8, 9, 14, 6, 5, 0, 4, 11, 7, 13, 12, 12, 14, 12, 14, 8, 3, 9, 6, 9, 5, 9, 14, 7, 11, 11, 13, 8, 13, 6, 10, 0, 11, 9, 11, 3, 10, 10, 13, 7, 1, 11, 2, 10, 13, 9, 9, 5, 0, 1, 11, 3, 14, 7, 11, 14, 4, 2, 1, 8, 5, 2, 11, 2, 10, 9, 3, 14, 4, 6, 11, 6, 0, 9, 9, 5, 3, 8, 5, 11, 1, 6, 6, 3, 3, 13, 3, 4, 2, 12, 0, 0, 2, 10, 1, 10, 10, 10, 3, 6, 10, 14, 8, 11, 4, 2, 13, 12, 0, 2, 11, 2, 2, 6, 3, 11, 7, 11, 4, 7, 1, 4, 0, 11, 6, 14, 10, 5, 12, 7, 6, 12, 7, 13, 13, 2, 4, 4, 5, 12, 0, 11, 6, 4, 4, 3, 1, 3, 0, 12, 0, 7, 0, 10, 3, 8, 14, 14, 8, 12, 2, 14, 1, 5, 12, 5, 5, 13, 7, 8, 14, 4, 8, 10, 10, 1, 5, 12, 8, 9, 8, 14, 7, 12, 13, 8, 12, 12, 4, 8, 3, 1, 6, 3, 5, 1, 7, 4, 13, 14, 5, 0, 4, 10, 5, 10, 1, 10, 9, 3, 9, 7, 3, 12, 5, 2, 14, 9, 2, 0, 1, 3, 14, 12, 7, 0, 10, 1, 14, 11, 1, 12, 13, 1, 1, 2, 9, 7, 14, 1, 5, 7, 11, 0, 8, 6, 4, 0, 6, 6, 4, 3, 7, 13, 13, 9, 7, 7, 1, 11, 10, 10, 6, 3, 3, 9, 3, 13, 5, 13, 1, 5, 6, 6, 6, 12, 3, 7, 4, 10, 2, 3, 6, 4, 4, 6, 5, 12, 0, 8, 3, 7, 8, 1, 10, 11, 1, 14, 0, 14, 11, 2, 6, 2, 8, 14, 4, 13, 0, 1, 8, 8, 8, 11, 10, 6, 5, 13, 10, 6, 2, 11, 8, 14, 2, 7, 4, 12, 8, 9, 4, 5, 2, 14, 5, 11, 9, 4, 6, 8, 14, 3, 5, 9, 7, 12, 0, 6, 2, 0, 9, 12, 3, 11, 10, 0, 2, 1, 9, 0, 13, 1, 12, 14, 1, 4, 3, 13, 7, 8, 6, 10, 12, 0, 10, 0, 10, 10, 0, 14, 4, 2, 11, 8, 7, 2, 1, 13, 13, 5, 8, 13, 13, 11, 8, 9, 5, 9, 13, 7, 2, 4, 4, 11, 4, 6, 13, 8, 2, 13, 7, 1]

sequence_groups1 = transform_data(data.process_scrambled(labels, ['math8.txt'], channels=channels, sample_rate=250,
                                                                 surrounding=surrounding, exclude=set([25, 99, 407])))

print len(sequence_groups1)
print map(len, sequence_groups1)

lengths = map(len, data.get_inputs(sequence_groups1)[0])
print min(lengths), np.mean(lengths), max(lengths)

#channels = range(0, 4)
channels = [0]

#labels = [1, 9, 7, 1, 13, 5, 14, 4, 2, 12, 9, 0, 10, 5, 7, 4, 9, 6, 3, 3, 9, 8, 7, 4, 12, 9, 3, 1, 7, 14, 9, 13, 6, 12, 11, 4, 5, 2, 5, 10, 11, 1, 11, 10, 5, 11, 1, 0, 4, 5, 3, 3, 11, 13, 10, 13, 12, 8, 12, 9, 1, 11, 12, 1, 0, 10, 0, 12, 11, 6, 5, 14, 13, 4, 6, 7, 11, 9, 10, 8, 1, 3, 3, 12, 3, 8, 11, 14, 2, 2, 6, 0, 13, 9, 10, 14, 8, 14, 10, 6, 1, 8, 4, 8, 5, 2, 7, 4, 2, 13, 12, 14, 4, 2, 7, 0, 14, 3, 14, 10, 4, 11, 1, 8, 0, 8, 6, 0, 0, 5, 2, 6, 6, 0, 8, 9, 6, 5, 10, 2, 2, 13, 7, 12, 7, 13, 13, 7, 14, 3]

labels = [14, 12, 5, 3, 1, 9, 2, 12, 2, 6, 0, 13, 12, 2, 13, 12, 9, 9, 10, 7, 12, 8, 7, 9, 1, 0, 14, 3, 9, 14, 11, 0, 7, 9, 5, 5, 4, 5, 8, 9, 14, 6, 5, 0, 4, 11, 7, 13, 12, 12, 14, 12, 14, 8, 3, 9, 6, 9, 5, 9, 14, 7, 11, 11, 13, 8, 13, 6, 10, 0, 11, 9, 11, 3, 10, 10, 13, 7, 1, 11, 2, 10, 13, 9, 9, 5, 0, 1, 11, 3, 14, 7, 11, 14, 4, 2, 1, 8, 5, 2, 11, 2, 10, 9, 3, 14, 4, 6, 11, 6, 0, 9, 9, 5, 3, 8, 5, 11, 1, 6, 6, 3, 3, 13, 3, 4, 2, 12, 0, 0, 2, 10, 1, 10, 10, 10, 3, 6, 10, 14, 8, 11, 4, 2, 13, 12, 0, 2, 11, 2, 2, 6, 3, 11, 7, 11, 4, 7, 1, 4, 0, 11, 6, 14, 10, 5, 12, 7, 6, 12, 7, 13, 13, 2, 4, 4, 5, 12, 0, 11, 6, 4, 4, 3, 1, 3, 0, 12, 0, 7, 0, 10, 3, 8, 14, 14, 8, 12, 2, 14, 1, 5, 12, 5, 5, 13, 7, 8, 14, 4, 8, 10, 10, 1, 5, 12, 8, 9, 8, 14, 7, 12, 13, 8, 12, 12, 4, 8, 3, 1, 6, 3, 5, 1, 7, 4, 13, 14, 5, 0, 4, 10, 5, 10, 1, 10, 9, 3, 9, 7, 3, 12, 5, 2, 14, 9, 2, 0, 1, 3, 14, 12, 7, 0, 10, 1, 14, 11, 1, 12, 13, 1, 1, 2, 9, 7, 14, 1, 5, 7, 11, 0, 8, 6, 4, 0, 6, 6, 4, 3, 7, 13, 13, 9, 7, 7, 1, 11, 10, 10, 6, 3, 3, 9, 3, 13, 5, 13, 1, 5, 6, 6, 6, 12, 3, 7, 4, 10, 2, 3, 6, 4, 4, 6, 5, 12, 0, 8, 3, 7, 8, 1, 10, 11, 1, 14, 0, 14, 11, 2, 6, 2, 8, 14, 4, 13, 0, 1, 8, 8, 8, 11, 10, 6, 5, 13, 10, 6, 2, 11, 8, 14, 2, 7, 4, 12, 8, 9, 4, 5, 2, 14, 5, 11, 9, 4, 6, 8, 14, 3, 5, 9, 7, 12, 0, 6, 2, 0, 9, 12, 3, 11, 10, 0, 2, 1, 9, 0, 13, 1, 12, 14, 1, 4, 3, 13, 7, 8, 6, 10, 12, 0, 10, 0, 10, 10, 0, 14, 4, 2, 11, 8, 7, 2, 1, 13, 13, 5, 8, 13, 13, 11, 8, 9, 5, 9, 13, 7, 2, 4, 4, 11, 4, 6, 13, 8, 2, 13, 7, 1]

sequence_groups2 = transform_data(data.process_scrambled(labels, ['math13.txt'], channels=channels, sample_rate=250,
                                                                   surrounding=surrounding, exclude=set([36, 66, 73])))

print len(sequence_groups2)
print map(len, sequence_groups2)

lengths = map(len, data.get_inputs(sequence_groups2)[0])
print min(lengths), np.mean(lengths), max(lengths)


length = 450
sequence_groups1 = data.transform.pad_truncate(sequence_groups1, length)
sequence_groups2 = data.transform.pad_truncate(sequence_groups2, length)

colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
colors = np.array(colors)[channels]
fig, axes = plt.subplots(len(sequence_groups1), 2, figsize=(8, 10))
plt.subplots_adjust(left=0.10, right=0.94, bottom=0.05, top=0.94)
axes[0][0].set_title('Session 1')
axes[0][1].set_title('Session 2')
for i in range(len(sequence_groups1)):
    avg1 = np.mean(sequence_groups1[i], axis=0)
    avg2 = np.mean(sequence_groups2[i], axis=0)
    axes[i][0].set_ylim((-35, 35))
    axes[i][1].set_ylim((-35, 35))
    axes[i][0].set_ylabel(str(i))
    for j in range(len(channels)):
        axes[i][0].plot(avg1[:,j], c=colors[j])
        axes[i][1].plot(avg2[:,j], c=colors[j])
plt.show()

