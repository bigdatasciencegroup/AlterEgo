import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

def process_file(filepath, sample_rate=250, channels=range(4, 8)):
    print 'Processing', filepath
    f = open(filepath, 'r')
    contents = map(lambda x: x.strip(), f.readlines())
    frames_original = filter(lambda x: x and x[0] != '%', contents)[1:]
    frames_original = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames_original)

    # assert equal data vector lengths
    assert len(set(map(len, frames_original))) == 1, map(len, frames_original)

    # (7 channels + digital trigger)
    frames = map(lambda a: map(float, a[1:9]) + [float(a[-3] if sample_rate==250 else a[-2])], frames_original)
    timestamps = map(lambda a: a[-1], frames_original)
    frames = np.array(frames)
    return frames[:,channels+[-1]]

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
#    
    return sequence_groups


original = process_file('patient_test/patient1/1.txt')
sequence = transform_data(original)
sequence[:,-1] = original[:,-1] * 100

print np.shape(sequence)

plt.plot(sequence)
plt.show()
