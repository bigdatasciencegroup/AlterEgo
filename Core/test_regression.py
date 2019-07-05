import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

filepath = 'data/data/161_012_30_trials.txt'
#filepath = 'data/data/167_silence_30_trials.txt'
sample_rate = 250

contents = map(lambda x: x.strip(), open(filepath, 'r').readlines())
frames = filter(lambda x: x and x[0] != '%', contents)[1:]
frames = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames)
sequence_data = []
for frame in frames:
    channel_data = [0.0] * 8
    for i in range(8):
        channel_data[i] = float(frame[i+1])
    sequence_data.append(channel_data)
sequence_data = np.array(sequence_data)

print np.shape(sequence_data)

indices = range(8600, 9600) # one

sequence_data = sequence_data[indices]

# Correct DC offset, drift, and amplitude
sequence_data = data.transform.subtract_initial(sequence_data)
sequence_data = data.transform.highpass_filter(sequence_data, 0.5, sample_rate)
sequence_data = data.transform.subtract_mean(sequence_data)

notch_filtered = sequence_data
notch1 = 59.9
notch2 = 119.8
notch_filtered = data.transform.notch_filter(notch_filtered, notch2, sample_rate)
notch_filtered = data.transform.notch_filter(notch_filtered, notch1, sample_rate)
notch_filtered = data.transform.notch_filter(notch_filtered, notch2, sample_rate)
notch_filtered = data.transform.notch_filter(notch_filtered, notch1, sample_rate)

std_normalized = data.transform.normalize_std(notch_filtered)

low = 1
high = 10
bandpass_filtered = data.transform.bandpass_filter(std_normalized, low, high, sample_rate)


sequence = bandpass_filtered[460:600]

def sinusoid_with_bias(x, t, y):
    return x[0] + x[1] * np.sin(x[2] * np.array(t) + x[3]) - y

def double_sinusoid_with_bias(x, t, y):
    t = np.array(t)
    return x[0] + x[1] * np.sin(x[2] * t + x[3]) \
            + x[4] * np.sin(x[5] * t + x[6]) - y

colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
for i in range(np.shape(sequence)[1]):
    plt.plot(sequence[:,i], c=colors[i], alpha=1.0)
for i in range(np.shape(sequence)[1]):
    seq = sequence[:,i]
    fn = sinusoid_with_bias
    initial = [0, 1, 0.1, 0]
#    fn = double_sinusoid_with_bias
#    initial = [0, 1, 0.1, 0, 1, 0.1, 0]
    res_lsq = least_squares(fn, initial, args=(range(len(seq)), seq))
    print i, res_lsq.x
    xs = np.arange(0, len(seq), 0.01)
    plt.plot(xs, map(lambda x: fn(res_lsq.x, x, 0.0), xs), '--', c=colors[i], lw=1)
plt.show()



