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


channels = range(1, 8)
#channels = range(4, 8)
#channels = [2, 4, 5, 6, 7]
surrounding = 250

labels = [2, 3, 4, 3, 0, 0, 4, 2, 4, 3, 3, 3, 2, 3, 1, 4, 4, 2, 2, 1, 4, 1, 4, 2, 2, 0, 3, 1, 3, 1, 2, 4, 3, 1, 2, 2, 0, 0, 4, 0, 1, 0, 1, 0, 1, 0, 0, 1, 4, 3]

sequence_groups1 = transform_data(data.process_scrambled(labels, ['doug.txt'], channels=channels, sample_rate=250,
                                                                 surrounding=surrounding, exclude=set([25, 99, 407])))

print len(sequence_groups1)
print map(len, sequence_groups1)

lengths = map(len, data.get_inputs(sequence_groups1)[0])
print min(lengths), np.mean(lengths), max(lengths)

length = 450
sequence_groups1 = data.transform.pad_truncate(sequence_groups1, length)

colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
colors = np.array(colors)[channels]
fig, axes = plt.subplots(len(sequence_groups1), 1, figsize=(8, 10))
plt.subplots_adjust(left=0.10, right=0.94, bottom=0.05, top=0.94)
axes[0].set_title('Session 1')
for i in range(len(sequence_groups1)):
    avg1 = np.mean(sequence_groups1[i], axis=0)
    axes[i].set_ylim((-100, 100))
    axes[i].set_ylabel(str(i))
    for j in range(len(channels)):
        axes[i].plot(avg1[:,j], c=colors[j])
plt.show()
