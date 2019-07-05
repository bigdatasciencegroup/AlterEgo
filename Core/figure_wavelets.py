import numpy as np
#from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data

filepath = 'data/data/161_012_30_trials.txt'
##filepath = 'data/data/167_silence_30_trials.txt'
##filepath = 'data/data/70_subvocal_digits_15_trials.txt'
filepath = 'data/data/168_012_1k_30.txt'
sample_rate = 1000 #250

contents = map(lambda x: x.strip(), open(filepath, 'r').readlines())
frames = filter(lambda x: x and x[0] != '%', contents)[1:]
frames = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames)
sequence_data = []
for frame in frames:
    channel_data = [0.0] * 2 # change to 8
    for i in range(len(channel_data)):
        channel_data[i] = float(frame[i+1+5]) # change to i+1
    channel_data[1] = 0.0 # REMOVE
    sequence_data.append(channel_data)
sequence_data = np.array(sequence_data)

print np.shape(sequence_data)

##indices = range(len(sequence_data)) # all
##indices = range(3000, 45000) # no edges
##indices = range(8000, 16000) # many
#indices = range(8000, 12500) # few
##indices = range(8600, 9600) # one

#indices = range(len(sequence_data)) # all
#indices = range(30000, 250000) # no edges
#indices = range(65000, 154000) # many
#indices = range(97000, 127000) # few
#indices = range(115000, 127000) # fewer
#indices = range(118000, 126000) # one
#indices = range(118000, 119000) # sub

#indices = range(97000, 113000) # select
indices = range(97000, 104000) # select


sequence_data = sequence_data[indices]

# Correct DC offset, drift, and amplitude
drift_low_frequency = 0.5 #0.5
sequence_data = data.transform.subtract_initial(sequence_data)
sequence_data = data.transform.highpass_filter(sequence_data, drift_low_frequency, sample_rate)
sequence_data = data.transform.subtract_mean(sequence_data)


notch_filtered = sequence_data
notch1 = 60 #59.9
num_times = 3 #2
freqs = map(int, map(round, np.arange(1, sample_rate/(2. * notch1)) * notch1))
for _ in range(num_times):
    for f in reversed(freqs):
        notch_filtered = data.transform.notch_filter(notch_filtered, f, sample_rate)


#std_normalized = data.transform.normalize_std(notch_filtered)


def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))

def gaussian_wavelet(n, sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return np.array([1.0 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r])

def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

#kernel = normalize_kernel(gaussian_wavelet(25), subtract_mean=True)
#kernel = normalize_kernel(np.arange(50, dtype=np.float32), subtract_mean=True)
#kernel = normalize_kernel(-np.sin(np.arange(250)/250. * 2*np.pi), subtract_mean=True)
period1 = 250 * sample_rate // 250
sin_kernel1 = normalize_kernel(np.sin(np.arange(period1)/float(period1) * 1*np.pi), subtract_mean=True)

gaussian_kernel = normalize_kernel(gaussian_wavelet(500, 100.0), subtract_mean=True)
#square_kernel = normalize_kernel([0.0]*130 + [1.0]*130 + [0.0]*130, subtract_mean=True)
square_kernel = normalize_kernel([0.0] + [1.0]*125 + [-1.0]*125 + [0], subtract_mean=True)
period2 = 100 * sample_rate // 250
sin_kernel2 = normalize_kernel(np.sin(np.arange(period2)/float(period2) * 2*np.pi), subtract_mean=True)
#plt.plot(gaussian); plt.plot(sin_kernel); plt.show()

ricker_width = 35 * sample_rate // 250
ricker_sigma = 4.0 * sample_rate / 250 #4.0...
ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))

#peak_kernel = normalize_kernel(gaussian_wavelet(15, 2.0), subtract_mean=True)
#plt.plot(peak_kernel); plt.show()

ricker_convolved = data.transform.correlate(notch_filtered, ricker_kernel)
#ricker_convolved = np.clip(data.transform.correlate(notch_filtered, ricker_kernel), 0, np.inf)
#ricker_convolved = data.transform.correlate(np.clip(data.transform.correlate(notch_filtered, ricker_kernel), 0, np.inf), ricker_kernel)
#ricker_convolved = data.transform.correlate(data.transform.correlate(notch_filtered, ricker_kernel) ** 2, ricker_kernel)
#ricker_convolved = data.transform.correlate(np.clip(data.transform.correlate(notch_filtered, ricker_kernel), 0, np.inf) ** 2, ricker_kernel)

#ricker_convolved = data.transform.correlate(notch_filtered, ricker_kernel) ** 2
#ricker_convolved = data.transform.correlate(data.transform.correlate(notch_filtered, ricker_kernel) ** 2, ricker_kernel)
#ricker_convolved = notch_filtered - 7 * data.transform.correlate(data.transform.correlate(notch_filtered, ricker_kernel) ** 2, ricker_kernel)

ricker_subtraction_multiplier = 2
ricker_subtracted = notch_filtered - ricker_subtraction_multiplier * ricker_convolved

use_ricker = True

low = 0.5 #0.5
high = 8 #8
order = 1
bandpass_filtered = data.transform.bandpass_filter(ricker_subtracted if use_ricker else notch_filtered,
                                                   low, high, sample_rate, order=order)

sin_convolved1 = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, sin_kernel1)
#sin_convolved2 = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, sin_kernel2)

average_width = 5 * sample_rate // 250
average_kernel = normalize_kernel([1.0] * average_width)
#window_averaged = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, average_kernel)

square_convolved = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, square_kernel)
gaussian_convolved = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, gaussian_kernel)



fft = data.transform.fft(bandpass_filtered)
#fft = data.transform.fft(notch_filtered)
fft_magnitudes = np.abs(fft) / len(sequence_data)
fft_phases = np.angle(fft)
fft_magnitudes = fft_magnitudes[:len(fft)/2+1]
fft_phases = fft_phases[:len(fft)/2+1]
fft_freq_display = int(min((len(fft)/2+1) * (sample_rate * 1.0/len(sequence_data)), float('inf')))
fft_y = fft_magnitudes[:fft_freq_display * len(sequence_data) / sample_rate]
fft_x = map(lambda x: x * (sample_rate * 1.0/len(sequence_data)), range(len(fft_y)))

low2 = low
high2 = high
fft = data.transform.fft(ricker_subtracted if use_ricker else notch_filtered)
low_index = int(low2 * len(sequence_data) / sample_rate)
high_index = int(high2 * len(sequence_data) / sample_rate)
fft[high_index:-high_index,:] = 0.0
if low_index:
    fft[:low_index,:] = 0.0
    fft[-low_index:,:] = 0.0
ifft = np.fft.ifft(fft, axis=0)
fft_modified_magnitudes = np.abs(fft) / len(sequence_data)
fft_modified_phases = np.angle(fft)
fft_modified_magnitudes = fft_modified_magnitudes[:len(fft)/2+1]
fft_modified_phases = fft_modified_phases[:len(fft)/2+1]
fft_modified_y = fft_modified_magnitudes[:fft_freq_display * len(sequence_data) / sample_rate]
fft_modified_x = map(lambda x: x * (sample_rate * 1.0/len(sequence_data)), range(len(fft_modified_y)))

#plt.plot(bandpass_filtered[:,4])
#plt.plot(np.fft.ifft(fft, axis=0)[:,4]-5)
#plt.show()



#def window_split(sequence, window, stride=None):
#    if stride is None: stride = window
#    return np.array([np.concatenate(sequence[i:i+window,:]) for i in range(0, len(sequence)-window+1, stride)])
#
#fragments = window_split(ricker_subtracted[:,4:], 250, 1)
#print np.shape(fragments)
#pca = PCA(n_components=2)
#reduced = pca.fit_transform(fragments)
#print np.shape(reduced)
#plt.scatter(*np.transpose(reduced), marker=',', s=(72./plt.gcf().dpi)**2)
#plt.show()
#
#
##kill
#
#
##for fragment in fragments:
##    plt.plot(np.reshape(fragment, [-1, 4]))
##    plt.show()





#colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown'] # Change back to
colors = ['red', 'gray', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown']
alpha = 1.0
lw = 1.0
sd = 2.5
fig = plt.figure(figsize=(12, 7))
gridspec.GridSpec(4, 5)

plt.subplot2grid((4, 5), (0, 1), colspan=4, rowspan=1)
plt.title('Processed Signal (Single Channel)')
for i in range(np.shape(bandpass_filtered)[1]):
    plt.plot(bandpass_filtered[:,i], c=colors[i], alpha=alpha, lw=lw)
plt.gca().set_xlim((0, len(bandpass_filtered)))
ylim = max(np.std(bandpass_filtered[len(bandpass_filtered)/10:len(bandpass_filtered)*9/10], axis=1)) * sd
#ax4.set_ylim((-ylim, ylim))
#axes[0].set_ylim(ax2.get_ylim())

plt.subplot2grid((4, 5), (1, 0), colspan=1, rowspan=1)
plt.title('Normalized Sine Wavelet\n(half period, period=2000)')
plt.plot(sin_kernel1)
#plt.gca().set_xlim((0, len(sin_kernel1)))

plt.subplot2grid((4, 5), (1, 1), colspan=4, rowspan=1)
plt.title('Sine Wavelet Convolution')
for i in range(np.shape(sin_convolved1)[1]):
    plt.plot(sin_convolved1[:,i], c=colors[i], alpha=alpha, lw=lw)
plt.gca().set_xlim((0, len(sin_convolved1)))
ylim = max(np.std(sin_convolved1[len(sin_convolved1)/10:len(sin_convolved1)*9/10], axis=1)) * sd
#ax5.set_ylim((-ylim, ylim))
#ax5.set_ylim(ax2.get_ylim())

plt.subplot2grid((4, 5), (2, 0), colspan=1, rowspan=1)
plt.title('Normalized Gaussian Wavelet\n(width=500, sigma=100)')
plt.plot(gaussian_kernel)
#plt.gca().set_xlim((0, len(gaussian_kernel)))

plt.subplot2grid((4, 5), (2, 1), colspan=4, rowspan=1)
plt.title('Gaussian Wavelet Convolution')
for i in range(np.shape(gaussian_convolved)[1]):
    plt.plot(gaussian_convolved[:,i], c=colors[i], alpha=alpha, lw=lw)
plt.gca().set_xlim((0, len(gaussian_convolved)))
ylim = max(np.std(gaussian_convolved[len(gaussian_convolved)/10:len(gaussian_convolved)*9/10], axis=1)) * sd
#axes[2].set_ylim((-ylim, ylim))

plt.subplot2grid((4, 5), (3, 0), colspan=1, rowspan=1)
plt.title('Normalized Square Wavelet\n(width=250)')
plt.plot(square_kernel)
#plt.gca().set_xlim((0, len(square_kernel)))

plt.subplot2grid((4, 5), (3, 1), colspan=4, rowspan=1)
plt.title('Square Wavelet Convolution')
for i in range(np.shape(square_convolved)[1]):
    plt.plot(square_convolved[:,i], c=colors[i], alpha=alpha, lw=lw)
plt.gca().set_xlim((0, len(square_convolved)))
ylim = max(np.std(square_convolved[len(square_convolved)/10:len(square_convolved)*9/10], axis=1)) * sd
#axes[2].set_ylim((-ylim, ylim))

plt.subplots_adjust(left=0.05, right=0.98, bottom=0.04, top=0.96, hspace=0.80, wspace=0.30)
plt.savefig('figure_wavelets.png')
plt.show()