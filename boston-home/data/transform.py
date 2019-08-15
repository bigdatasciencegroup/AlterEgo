import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, iirnotch, lfilter
from scipy.signal import convolve as _convolve
from scipy.ndimage.filters import gaussian_filter1d

apply_subtract_initial = True
apply_subtract_mean = False
apply_normalize_std = False
apply_notch_filter = True
apply_highpass_filter = False
apply_lowpass_filter = False
apply_bandpass_filter = True
apply_gaussian_filter = False

# Deprecated
def default_transform(data, apply_stretch_compress=False, apply_pad_truncate=False, length=None,
                      apply_subtract_initial=apply_subtract_initial, apply_subtract_mean=apply_subtract_mean,
                      apply_normalize_std=apply_normalize_std, apply_notch_filter=apply_notch_filter,
                      apply_highpass_filter=apply_highpass_filter, apply_lowpass_filter=apply_lowpass_filter,
                      apply_bandpass_filter=apply_bandpass_filter, apply_gaussian_filter=apply_gaussian_filter,
                      modify=False):
    if apply_stretch_compress or apply_pad_truncate:
        assert length > 0, 'Must provide a positive length'
        
    operations = [(apply_subtract_initial, lambda x, modify: subtract_initial(x, modify=modify)),
                  (apply_subtract_mean, lambda x, modify: subtract_mean(x, modify=modify)),
                  (apply_normalize_std, lambda x, modify: normalize_std(x, modify=modify)),
                  (apply_notch_filter, lambda x, modify: notch_filter(x, 60, 250, modify=modify)),
                  (apply_highpass_filter, lambda x, modify: highpass_filter(x, 0.1, 250, modify=modify)),
                  (apply_lowpass_filter, lambda x, modify: lowpass_filter(x, 10, 250, modify=modify)),
                  (apply_bandpass_filter, lambda x, modify: bandpass_filter(x, 1, 10, 250, modify=modify)),
                  (apply_gaussian_filter, lambda x, modify: gaussian_filter(x, 3, modify=modify)), #3
                  (apply_stretch_compress, lambda x, modify: stretch_compress(x, length, modify=modify)),
                  (apply_pad_truncate, lambda x, modify: pad_truncate(x, length, position=0.5, modify=modify))]
    
    operations = filter(lambda x: x[0], operations)
    for i in range(len(operations)):
        data = operations[i][1](data, modify or i > 0)
        
    return data

def default_transform_new(data, sample_rate=250, **kwargs):
    drift_low_frequency = 0.5 #0.5
    data = subtract_initial(data)
    data = highpass_filter(data, drift_low_frequency, sample_rate)
    data = subtract_mean(data)

    notch1 = 60 #59.9
    num_times = 3 #2
    freqs = map(int, map(round, np.arange(1, sample_rate/(2. * notch1)) * notch1))
    for _ in range(num_times):
        for f in reversed(freqs):
            data = notch_filter(data, f, sample_rate)

    #sequence_groups = data.transform.normalize_std(sequence_groups)

    def normalize_kernel(kernel, subtract_mean=False):
        if subtract_mean:
            kernel = np.array(kernel, np.float32) - np.mean(kernel)
        return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
    def ricker_function(t, sigma):
        return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
    def ricker_wavelet(n, sigma):
        return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250 #4.0...
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))

    ricker_convolved = correlate(data, ricker_kernel)
    ricker_subtraction_multiplier = 2
    data = data - ricker_subtraction_multiplier * ricker_convolved

#    period = 250 * sample_rate // 250
#    sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
#    data = correlate(data, sin_kernel)

    low = 0.5 #0.5
    high = 8 #8
    order = 1
    
    data = bandpass_filter(data, low, high, sample_rate, order=order)
    
#    data = fft(data)
#    data = fft_frequency_cutoff(data, low, high, sample_rate)
#    data = np.real(ifft(data))
    
    return data

def augment(data, arg_sets, include_original=False):
    axis = augment_axis(data)
    assert axis == 2
    if include_original:
        arg_sets.insert(0, (lambda x, modify: x, [], {}))
    augmented = np.array(map(lambda x: x[0](data, *x[1], modify=False, **x[2]), arg_sets))
    sequence_groups = []
    for i in range(np.shape(augmented)[1]):
        sequence_groups.append(np.concatenate(augmented[:,i], axis=0))
    return np.array(sequence_groups)

def augment_axis(data, axis=0):
    return axis if type(data[0][0]) in [np.float64, float] else augment_axis(data[0], axis+1)

def augment_pad_truncate_intervals(data, length, intervals):
    if intervals <= 1:
        return pad_truncate(data, length)
    return augment(data, [(pad_truncate, [length], dict(position=pos)) \
                          for pos in np.arange(0, 1+1./(intervals-1), 1./(intervals-1))])

def apply_recursively(data, fn, modify):
    if type(data[0][0]) in [np.float32, np.float64, float, np.complex128]:
        return fn(data)
    if modify:
        for i in range(len(data)):
            data[i] = apply_recursively(data[i], fn, modify)
        return data
    return np.array(map(lambda x: apply_recursively(x, fn, modify), data))

def channel_specific(seq, fn, modify, *args, **kwargs):
    result = seq if modify else np.zeros(np.shape(seq))
    for c in range(np.shape(seq)[1]):
        result[:,c] = fn(seq[:,c], *args, **kwargs)
    return result

def numpify(data, modify=False):
    return apply_recursively(data, np.array, modify)

# Subtract mean from signals
def subtract_mean(data, modify=False):
    return apply_recursively(data, lambda seq: seq - np.mean(seq, axis=0), modify)

# Subtract initial from signals
def subtract_initial(data, modify=False):
    return apply_recursively(data, lambda seq: seq - seq[0,:], modify)

# Normalize by dividing by standard deviation
def normalize_std(data, modify=False):
    def nan_to_zeros(s):
        s[np.where(np.isnan(s))] = 0.0
        return s
    data = apply_recursively(data, lambda seq: seq / np.std(seq, axis=0), modify)
    data = apply_recursively(data, nan_to_zeros, modify)
    return data

# Apply bandpass filter
def _bandpass_filter_channel(x, low, high, fs, order):
    nyq = 0.5 * fs
    l = low / nyq
    h = high / nyq
    b, a = butter(order, [l, h], btype='band')
    y = lfilter(b, a, x)
    return y
def bandpass_filter(data, low, high, fs, order=1, modify=False):
    return apply_recursively(data, lambda seq:
                             channel_specific(seq, _bandpass_filter_channel, modify, low, high, fs, order), modify)

# Apply highpass filter
def _highpass_filter_channel(x, low, fs, order):
    nyq = 0.5 * fs
    l = low / nyq
    b, a = butter(order, l, btype='high')
    y = lfilter(b, a, x)
    return y
def highpass_filter(data, low, fs, order=1, modify=False):
    return apply_recursively(data, lambda seq:
                             channel_specific(seq, _highpass_filter_channel, modify, low, fs, order), modify)

# Apply lowpass filter
def _lowpass_filter_channel(x, high, fs, order):
    nyq = 0.5 * fs
    h = high / nyq
    b, a = butter(order, h, btype='low')
    y = lfilter(b, a, x)
    return y
def lowpass_filter(data, high, fs, order=1, modify=False):
    return apply_recursively(data, lambda seq:
                             channel_specific(seq, _lowpass_filter_channel, modify, high, fs, order), modify)

# Apply notch filter
def _notch_filter_channel(x, freq, fs, Q=30):
    f = freq / (0.5 * fs)
    b, a = iirnotch(f, Q)
    y = lfilter(b, a, x)
    return y
def notch_filter(data, freq, fs, modify=False):
    return apply_recursively(data, lambda seq:
                             channel_specific(seq, _notch_filter_channel, modify, freq, fs), modify)

# Apply correlatation/convolution
def correlate(data, kernel, mode='same', modify=False):
    return apply_recursively(data, lambda seq:
                             channel_specific(seq, _convolve, modify, kernel[::-1], mode=mode), modify)

# Compress/expand sequences to average length
def stretch_compress(data, length, modify=False):
    return apply_recursively(data, lambda seq:
                             interp1d(range(len(seq)), seq, axis=0)(np.linspace(0, len(seq)-1, length)), modify)

# Compress/expand sequences to average length
def pad_truncate(data, length, position=0.5, modify=False):
    # if position = 1.0, it'll always have the signal on the right end, so pad and truncate from the left
    # if position = 0.0, it'll always have the signal on the left end, so pad and truncate from the right
    position = max(0, min(position, 1))
    left_size_p = lambda seq: max(0, int((length-len(seq)) * position))
    left_size_t = lambda seq: max(0, int((len(seq)-length) * position))
    return apply_recursively(data, lambda seq:
                             np.pad(seq, ((left_size_p(seq), max(0, int(length)-len(seq))-left_size_p(seq)),(0,0)),
                                    mode='constant')[left_size_t(seq):left_size_t(seq)+length,:], modify)

# Pad extra amount of space to each side
def pad_extra(data, length, modify=False):
    return apply_recursively(data, lambda seq: np.pad(seq, ((length, length), (0,0)), mode='constant'), modify)

# Apply gaussian 1d filter
def gaussian_filter(data, sigma, order=0, mode='nearest', modify=False):
    return apply_recursively(data, lambda seq:
                             gaussian_filter1d(seq, sigma, order=order, mode=mode, axis=0), modify)

# Apply FFT
def fft(data, modify=False):
    return apply_recursively(data, lambda seq: np.fft.fft(seq, axis=0), modify)

# Apply inverse FFT
def ifft(data, modify=False):
    return apply_recursively(data, lambda seq: np.fft.ifft(seq, axis=0), modify)

# Apply frequency cutoff
def fft_frequency_cutoff(data, low, high, sample_rate, modify=False):
    def cutoff(fft):
        low_index = int(low * len(fft) / sample_rate);
        high_index = int(high * len(fft) / sample_rate)
        fft[high_index:-high_index,:] = 0.0
        if low_index:
            fft[:low_index,:] = 0.0
            fft[-low_index:,:] = 0.0
        return fft
    return apply_recursively(data, lambda fft: cutoff(fft), modify)

# Retrieve real component
def real(data, modify=False):
    return apply_recursively(data, np.real, modify)

# Apply downsampling
def downsample(data, factor, modify=False):
    return apply_recursively(data, lambda x: x[::4,:], modify)
