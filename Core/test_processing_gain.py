import numpy as np
#from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

filepath = 'data/data/OpenBCI-RAW-2019-01-15_16-27-10.txt'
#filepath = 'data/data/OpenBCI-RAW-2019-01-15_16-31-30.txt'
#filepath = 'data/data/OpenBCI-RAW-2019-01-15_16-33-59.txt'
sample_rate = 1000 #250

contents = map(lambda x: x.strip(), open(filepath, 'r').readlines())
frames = filter(lambda x: x and x[0] != '%', contents)[1:]
frames = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames)
sequence_data = []
for frame in frames:
    channel_data = [0.0] * 2 # num channels
    for i in range(len(channel_data)):
        channel_data[i] = float(frame[i+1])
    sequence_data.append(channel_data)
sequence_data = np.array(sequence_data)

print np.shape(sequence_data)


#indices = range(len(sequence_data)) # all
#indices = range(44000, 112000) # a lot
#indices = range(51000, 67000) # many
indices = range(54000, 61000) # few

#indices = range(len(sequence_data)) # all
#indices = range(43000, 74000) # a lot
#indices = range(56000, 68000) # many

#indices = range(len(sequence_data)) # all
#indices = range(23000, 82000) # a lot
#indices = range(48000, 63000) # many


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

#gaussian_kernel = normalize_kernel(gaussian_wavelet(250, 50.0), subtract_mean=True)
#square_kernel = normalize_kernel([0.0]*30 + [1.0]*30 + [0.0]*30, subtract_mean=True)
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
sin_convolved1 = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, sin_kernel1)
#sin_convolved2 = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, sin_kernel2)

average_width = 5 * sample_rate // 250
average_kernel = normalize_kernel([1.0] * average_width)
window_averaged = data.transform.correlate(ricker_subtracted if use_ricker else notch_filtered, average_kernel)


low = 0.5 #0.5
high = 8 #8
order = 1
bandpass_filtered = data.transform.bandpass_filter(ricker_subtracted if use_ricker else notch_filtered,
                                                   low, high, sample_rate, order=order)


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



#colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
colors = ['red', 'gray', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown']
alpha = 1.0
lw = 1.0
sd = 2.5
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(18, 10))

ax1.set_title('1. Mostly Unprocessed (corrected for DC offset and drift)')
for i in range(np.shape(sequence_data)[1]):
    ax1.plot(sequence_data[:,i], c=colors[i], alpha=alpha, lw=lw)
ax1.set_xlim((0, len(sequence_data)))
ylim = max(np.std(sequence_data[len(sequence_data)/10:len(sequence_data)*9/10], axis=1)) * sd
#ylim = limit
ax1.set_ylim((-ylim, ylim))

ax2.set_title('2. Notch Filtered ('+str(list(freqs))+' Hz, x'+str(num_times)+') of Mostly Unprocessed')
for i in range(np.shape(notch_filtered)[1]):
    ax2.plot(notch_filtered[:,i], c=colors[i], alpha=alpha, lw=lw)
ax2.set_xlim((0, len(notch_filtered)))
ylim = max(np.std(notch_filtered[len(notch_filtered)/10:len(notch_filtered)*9/10], axis=1)) * sd
#ylim = limit
print ylim
ax2.set_ylim((-ylim, ylim))

#ax3.set_title('3. Standard Deviation Normalized of Notch Filtered')
#for i in range(np.shape(notch_filtered)[1]):
#    ax3.plot(notch_filtered[:,i], c=colors[i], alpha=alpha, lw=lw)
#ax3.set_xlim((0, len(notch_filtered)))
#ylim = max(np.std(notch_filtered[len(notch_filtered)/10:len(notch_filtered)*9/10], axis=1)) * sd
#ax3.set_ylim((-ylim, ylim))

ax3.set_title('3. Ricker Wavelet Convolution (sigma='+str(ricker_sigma)+') of Notch Filtered')
for i in range(np.shape(ricker_convolved)[1]):
    ax3.plot(ricker_convolved[:,i], c=colors[i], alpha=alpha, lw=lw)
ax3.set_xlim((0, len(ricker_convolved)))
ylim = max(np.std(ricker_convolved[len(ricker_convolved)/10:len(ricker_convolved)*9/10], axis=1)) * sd
#ylim = limit
ax3.set_ylim((-ylim, ylim))

ax4.set_title('4. Ricker-Subtracted (mult='+str(ricker_subtraction_multiplier)+') of Notch Filtered')
for i in range(np.shape(ricker_subtracted)[1]):
    ax4.plot(ricker_subtracted[:,i], c=colors[i], alpha=alpha, lw=lw)
ax4.set_xlim((0, len(ricker_subtracted)))
ylim = max(np.std(ricker_subtracted[len(ricker_subtracted)/10:len(ricker_subtracted)*9/10], axis=1)) * sd
#ax4.set_ylim((-ylim, ylim))
ax4.set_ylim(ax2.get_ylim())

#ax5.set_title('5. Sine Wavelet Convolution (full period, width='+str(period2)+') of ' + ('Ricker-Subtracted' if use_ricker else 'Notch Filtered'))
#for i in range(np.shape(sin_convolved2)[1]):
#    ax5.plot(sin_convolved2[:,i], c=colors[i], alpha=alpha, lw=lw)
#ax5.set_xlim((0, len(sin_convolved2)))
#ylim = max(np.std(sin_convolved2[len(sin_convolved2)/10:len(sin_convolved2)*9/10], axis=1)) * sd
#ax5.set_ylim((-ylim, ylim))

ax5.set_title('5. Sliding Window Average (width='+str(average_width)+') of ' + ('Ricker-Subtracted' if use_ricker else 'Notch Filtered'))
for i in range(np.shape(window_averaged)[1]):
    ax5.plot(window_averaged[:,i], c=colors[i], alpha=alpha, lw=lw)
ax5.set_xlim((0, len(window_averaged)))
ylim = max(np.std(window_averaged[len(window_averaged)/10:len(window_averaged)*9/10], axis=1)) * sd
#ax5.set_ylim((-ylim, ylim))
ax5.set_ylim(ax2.get_ylim())

ax6.set_title('6. Sine Wavelet Convolution (half period, width='+str(period1)+') of ' + ('Ricker-Subtracted' if use_ricker else 'Notch Filtered'))
for i in range(np.shape(sin_convolved1)[1]):
    ax6.plot(sin_convolved1[:,i], c=colors[i], alpha=alpha, lw=lw)
ax6.set_xlim((0, len(sin_convolved1)))
ylim = max(np.std(sin_convolved1[len(sin_convolved1)/10:len(sin_convolved1)*9/10], axis=1)) * sd
#ylim = limit
ax6.set_ylim((-ylim, ylim))

ax7.set_title('7. Bandpass Filtered ('+str(low)+'-'+str(high)+'Hz, order '+str(order)+') of ' + ('Ricker-Subtracted' if use_ricker else 'Notch Filtered'))
for i in range(np.shape(bandpass_filtered)[1]):
    ax7.plot(bandpass_filtered[:,i], c=colors[i], alpha=alpha, lw=lw)
ax7.set_xlim((0, len(bandpass_filtered)))
ylim = max(np.std(bandpass_filtered[len(bandpass_filtered)/10:len(bandpass_filtered)*9/10], axis=1)) * sd
#ax7.set_ylim((-ylim, ylim))
ax7.set_ylim(ax2.get_ylim())

ax8.set_title('8. Frequency Domain ($\leq$'+str(fft_freq_display)+'Hz) of Bandpass Filtered')
for i in range(np.shape(fft_y)[1]):
    ax8.plot(fft_x, fft_y[:,i], c=colors[i], alpha=alpha, lw=lw)
ax8.set_xlim((drift_low_frequency, max(fft_x)))
ax8.set_xscale('log')

ax9.set_title('9. Inverse FFT of Modified Frequency Domain (only '+str(low2)+'-'+str(high2)+'Hz) of ' + ('Ricker-Subtracted' if use_ricker else 'Notch Filtered'))
for i in range(np.shape(ifft)[1]):
    ax9.plot(ifft[:,i], c=colors[i], alpha=alpha, lw=lw)
ax9.set_xlim((0, len(ifft)))
ylim = max(np.std(ifft[len(ifft)/10:len(ifft)*9/10], axis=1)) * sd
#ax7.set_ylim((-ylim, ylim))
ax9.set_ylim(ax2.get_ylim())

ax10.set_title('10. Modified Frequency Domain ($\leq$'+str(fft_freq_display)+'Hz)')
for i in range(np.shape(fft_modified_y)[1]):
    ax10.plot(fft_modified_x, fft_modified_y[:,i], c=colors[i], alpha=alpha, lw=lw)
ax10.set_xlim((drift_low_frequency, max(fft_modified_x)))
ax10.set_xscale('log')

plt.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.95, hspace=0.70, wspace=0.10)
plt.show()