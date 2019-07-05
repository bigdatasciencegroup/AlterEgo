import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

def transform_data_plus_ricker(sequence_groups, sample_rate=1000):
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
#    sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

    #### Apply sine wavelet kernel
    #period = int(sample_rate)
    #sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
    #sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5 #0.5
    high_freq = 8 #8
    order = 1

    #### Apply soft bandpassing
#    sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)
    
    #### Apply hard bandpassing
    #sequence_groups = data.transform.fft(sequence_groups)
    #sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    #sequence_groups = np.real(data.transform.ifft(sequence_groups))
#    
    return sequence_groups, ricker_convolved

        
#### Load data
# sequence_groups[i] contains data for class i
# 4-dimensional data structure: (class, sequence_num, timestep, channel_num)

sequence_groups = data.words_10_20_sentences_dataset(channels=range(0, 8))
sequences = data.get_inputs(sequence_groups)[0]
sequences, ricker_sequences = transform_data_plus_ricker(sequences)

def collapse(indices):
    last = -2
    start = None
    collapsed = []
    for i in range(len(indices)):
        if indices[i] != last + 1:
            if start:
                collapsed.append((start + indices[i-1]) // 2)
            start = indices[i]
        last = indices[i]
    if start:
        collapsed.append((start + indices[-1]) // 2)
    return collapsed

stds = []
means = []
xs = np.arange(3, 30, 0.01)
for channel in range(8):
    print channel
    stds.append([])
    means.append([])
    for x in xs:
        diffs = []
        for ricker_sequence in ricker_sequences:
            indices = list(np.where(ricker_sequence[:,channel] > x)[0])
            collapsed = collapse(indices)
            diff = [collapsed[i+1] - collapsed[i] for i in range(len(collapsed)-1)]
            diffs += diff
        mean = np.mean(diff)
        mean = (diff[0] if len(diff) else 0) if np.isnan(mean) else mean
        std = np.std(diff)
        std = np.inf if np.isnan(std) or std == 0.0 else std
        stds[-1].append(std)
        means[-1].append(mean)
        print x, '\t', mean, '\t', std
    print
    
plt.plot(xs, np.transpose(stds))
plt.gca().set_xlim((min(xs), max(xs)))
plt.figure()
plt.plot(xs, np.transpose(means))
plt.gca().set_xlim((min(xs), max(xs)))
plt.show()

#    plt.plot(sequences[0][:,channel])
#    plt.plot(ricker_sequences[0][:,channel])
#    plt.show()
