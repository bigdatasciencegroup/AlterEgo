import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import os

import data

# channels = [1, 3, 4, 6] # Must be same as trained model if test_model==True
#channels = [1, 3, 4] # DO NOT CHANGE
channels = range(0,8)

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
#    period = int(sample_rate)
#    sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
#    sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5 #0.5
    high_freq = 8 #8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)
    
    #### Apply hard bandpassing
#    sequence_groups = data.transform.fft(sequence_groups)
#    sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
#    sequence_groups = np.real(data.transform.ifft(sequence_groups))
    
    return sequence_groups

word_map = ['hello there good morning', 'thank you i appreciate it', 'goodbye see you later', 'it was nice meeting you', 'wish you luck and success', 'how are you doing today', 'i want to sleep now', 'can you please help me', 'i am very hungry', 'going to the bathroom', 'you are welcome', 'super tired already', 'i have been doing good', 'what is your name', 'i feel sorry for that', 'Finished']
'''
Sara, Paul, George
1) [6, 6, 2, 5, 5, 3, 6, 3, 4, 5, 3, 2, 8, 3, 7, 0, 2, 4, 3, 8, 3, 2, 3, 1, 5]
2) [8, 6, 9, 2, 6, 0, 0, 0, 8, 1, 7, 8, 6, 3, 4, 1, 1, 2, 1, 4, 5, 7, 0, 8, 5]
3) [1, 9, 0, 4, 8, 0, 1, 5, 3, 2, 0, 1, 6, 7, 9, 5, 2, 4, 4, 9, 9, 9, 9, 7, 8]
4) [9, 0, 4, 2, 7, 5, 7, 1, 6, 9, 5, 8, 2, 8, 0, 7, 6, 1, 3, 9, 6, 7, 4, 7, 4]
5) [13, 11, 10, 11, 12, 12, 13, 11, 12, 14, 13, 10, 11, 14, 13, 12, 14, 10, 13, 14, 10, 10, 12, 11, 12]
6) [10, 13, 13, 13, 10, 10, 12, 10, 14, 11, 12, 12, 14, 11, 13, 11, 11, 14, 14, 14, 11, 13, 10, 12, 14]

Richard, Tina, Carol
1) [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2) [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
3) [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
4) [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
5) [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
6) [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
7) [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
8) [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
9) [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
10) [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
11) [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
12) [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
13) [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
14) [13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
15) [14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
'''

labels = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
labels = labels + [-1]

def on_data(history, trigger_history, index_history, count, samples_per_update, recorded_count):
    print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i+1) for i in range(8)])
    print str('{:.1f}'.format(count/250.)) + 's\t\t' + '\t'.join(
        map(lambda (i, x): '{:f}'.format(x) if i in channels else '--\t', enumerate(history[-1])))
    print
    print 'Sample #' + str(recorded_count + 1) + '/' + str(len(labels) - 1), '\t', word_map[labels[recorded_count]]
    print

data.serial.start('/dev/tty.usbserial-DM01HUN9',
                  on_data, channels=channels, transform_fn=transform_data,
                  history_size=1500, shown_size=1200, override_step=40)#35