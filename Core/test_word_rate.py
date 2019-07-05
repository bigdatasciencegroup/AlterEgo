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


sequence_groups1 = transform_data(data.words_10_20_sentences_dataset(include_surrounding=True, surrounding=500))
sequences1 = data.get_inputs(sequence_groups1)[0]

sequence_groups2 = transform_data(data.digits_sequences_session_2_dataset(include_surrounding=True, surrounding=500, channels=range(4, 8)))
sequences2 = data.get_inputs(sequence_groups2)[0]

lengths1 = map(len, sequences1)
lengths2 = map(len, sequences2)

sequences1 = data.transform.pad_truncate(sequences1, np.mean(lengths1))
sequences2 = data.transform.pad_truncate(sequences2, np.mean(lengths2))

plt.plot(np.mean(sequences1, axis=0))
plt.figure()
plt.plot(np.mean(sequences2, axis=0))
plt.show()

#avg_length1 = np.mean(np.array(lengths1) / 5.0)
#avg_length2 = np.mean(np.array(lengths2) / 3.0)
#avg_length_total = np.mean(np.concatenate([np.array(lengths1) / 5.0, np.array(lengths2) / 3.0]))
#
#print avg_length1
#print avg_length2
#print avg_length_total
