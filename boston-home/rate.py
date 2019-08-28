import numpy as np
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import os
from sklearn.utils.class_weight import compute_class_weight
from display_utils import DynamicConsoleTable
import math
from math import log
import time
import os.path
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import data

abs_path = os.path.abspath(os.path.dirname(__file__))

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

        
#### Load data
def dataset(**kwargs):
    patient_dir = 'patient_data/tina'
#    patient_dir = 'patient1'
    files = map(lambda x: patient_dir + '/' + x, filter(lambda x: '.txt' in x, os.listdir(patient_dir)))
    files.sort()
    print files
    return data.join([data.process(1, [file], **kwargs) for file in files])

# channels = range(1, 8) # DO NOT CHANGE
channels = range(0, 8)

total_data = dataset(channels=channels, surrounding=0)
print np.array(total_data).shape # (Files, Samples per file)
sequence_groups = transform_data(total_data)
print len(sequence_groups) # no. of Files
print map(len, sequence_groups) # List of samples per file
print np.array(sequence_groups).shape # (15,10)
print
wpm = 0
word_map = ['hello there good morning', 'thank you i appreciate it', 'goodbye see you later', 'it was nice meeting you', 'wish you luck and success', 'how are you doing today', 'i want to sleep now', 'can you please help me', 'i am very hungry', 'going to the bathroom', 'you are welcome', 'super tired already', 'i have been doing good', 'what is your name', 'i feel sorry for that']
for x, word in zip(sequence_groups, word_map):
    length = map(len, x)
    # print length
    num_words = len(word.split())
    for l in length:    wpm += float(num_words)/l

wpm = (60*250./150)*wpm
print 'WPM: {:.2f} per minute'.format(wpm)

N = 15
P = 0.77
bit_rate = wpm * (log(N,2) + P*log(P,2) + (1-P)*log((1-P)/(N-1),2))
print 'Bit-rate: {:.2f} bits per minute'.format(bit_rate)
