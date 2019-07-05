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

labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]

sequence_groups1 = transform_data(data.process_scrambled(labels, ['eric1.txt'], channels=channels, sample_rate=250))
print map(len, sequence_groups1)

lengths = map(len, data.get_inputs(sequence_groups1)[0])
print min(lengths), np.mean(lengths), max(lengths)

labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]

sequence_groups2 = transform_data(data.process_scrambled(labels, ['eric2.txt'], channels=channels, sample_rate=250))
print map(len, sequence_groups2)

lengths = map(len, data.get_inputs(sequence_groups2)[0])
print min(lengths), np.mean(lengths), max(lengths)


length = 450
sequence_groups1 = data.transform.pad_truncate(sequence_groups1, length)
sequence_groups2 = data.transform.pad_truncate(sequence_groups2, length)

colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
colors = np.array(colors)[channels]
fig, axes = plt.subplots(5, 2, figsize=(8, 10))
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.05, top=0.94)
axes[0][0].set_title('Session 1')
axes[0][1].set_title('Session 2')
for i in range(len(sequence_groups1)):
    avg1 = np.mean(sequence_groups1[i], axis=0)
    avg2 = np.mean(sequence_groups2[i], axis=0)
    axes[i][0].set_ylim((-35, 35))
    axes[i][1].set_ylim((-35, 35))
    for j in range(len(channels)):
        axes[i][0].plot(avg1[:,j], c=colors[j])
        axes[i][1].plot(avg2[:,j], c=colors[j])
plt.show()

