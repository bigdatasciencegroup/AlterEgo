import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

#sequence_groups = data.silence_and_not_silence_dataset()

#sequence_groups = data.process(1, ['data/test_data/testing_6_s2.txt'])
sequence_groups = data.process(1, ['data/data/OpenBCI-RAW-2018-10-23_09-15-13.txt'])
#sequence_groups = data.process(10, ['data/data/73_subvocal_digits_6_trials.txt'])
#sequence_groups = data.process(1, ['data/data/74_subvocal_silence_300_trials.txt'])
#sequence_groups = data.digits_dataset()

print np.shape(sequence_groups)

length = 300

sequence_groups = data.transform.default_transform(sequence_groups)
sequence_groups = data.transform.pad_truncate(sequence_groups, length)

#sequence_groups = data.transform.pad_extra(sequence_groups, length)
#sequence_groups = data.transform.augment_pad_truncate_intervals(sequence_groups, length, 20)

#for i in range(20):
#    plt.plot(sequence_groups[0][len(sequence_groups[0])/20*i+1][:,0])
#    plt.show()
#    
#1/0

#print np.shape(sequence_groups)

color_map = ['purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']

def plot_channel_across_trials_for_single_digit(digit=0, channel=1, show=True):
    plt.figure()
    min_v = [float('inf')]
    max_v = [-float('inf')]
    i = 0
    for sequence in sequence_groups[digit]:
        plt.plot(sequence[:,channel], alpha=0.5)
        tmp_min = (np.min(sequence[:,channel]), i, np.argmin(sequence[:,channel]))
        tmp_max = (np.max(sequence[:,channel]), i, np.argmax(sequence[:,channel]))
        min_v = min_v if min_v[0] < tmp_min[0] else tmp_min
        max_v = max_v if max_v[0] > tmp_max[0] else tmp_max
        i += 1
    print
    print 'min', min_v
    print list(sequence_groups[digit][min_v[1]][0,:])
    print 'max', max_v
    print list(sequence_groups[digit][max_v[1]][0,:])
    title = 'Digit ' + str(digit) + '  Channel ' + str(channel) + ' (' + color_map[channel] + ')'
    plt.title(title)
    if show: plt.show()
    
def plot_channels_for_single_digit(digit=0, show=True):
    plt.figure()
    tmp = np.mean(sequence_groups[digit], axis=0)
    for i in range(np.shape(sequence_groups[0][0])[1]):
        plt.plot(tmp[:,i], c=color_map[i])
    title = 'Digit ' + str(digit)
    plt.title(title)
    if show: plt.show()

#plot_channel_across_trials_for_single_digit(digit=5, channel=4) # good example
#plot_channel_across_trials_for_single_digit(digit=3, channel=4) # good example
#plot_channel_across_trials_for_single_digit(digit=3, channel=2) # good example
#plot_channel_across_trials_for_single_digit(digit=3, channel=3) # good example
#plot_channel_across_trials_for_single_digit(digit=3, channel=5) # good example
#plot_channel_across_trials_for_single_digit(digit=6, channel=1) # good example
#plot_channel_across_trials_for_single_digit(digit=6, channel=4) # good example

#for i in range(10):
#    for j in range(7):
#        plot_channel_across_trials_for_single_digit(digit=i, channel=j)

#for i in range(10):
#    plot_channels_for_single_digit(digit=i, show=False)
#plt.show()




plot_channel_across_trials_for_single_digit(digit=0, channel=0)
plot_channel_across_trials_for_single_digit(digit=0, channel=1)
plot_channel_across_trials_for_single_digit(digit=0, channel=2)
plot_channel_across_trials_for_single_digit(digit=0, channel=3)
plot_channel_across_trials_for_single_digit(digit=0, channel=4)
plot_channel_across_trials_for_single_digit(digit=0, channel=5)
plot_channel_across_trials_for_single_digit(digit=0, channel=6)


