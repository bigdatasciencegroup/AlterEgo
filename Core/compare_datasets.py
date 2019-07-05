import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

#sequence_groups = data.silence_and_not_silence_dataset()

sequence_groups1 = data.digits_session_2_dataset()
sequence_groups2 = data.join([
        data.process(1, ['data/test_data/testing_0_s2.txt']),
        data.process(1, ['data/test_data/testing_1_s2.txt']),
        data.process(1, ['data/test_data/testing_2_s2.txt']),
        data.process(1, ['data/test_data/testing_3_s2.txt']),
        data.process(1, ['data/test_data/testing_4_s2.txt']),
        data.process(1, ['data/test_data/testing_5_s2.txt']),
        data.process(1, ['data/test_data/testing_6_s2.txt']),
        data.process(1, ['data/test_data/testing_7_s2.txt']),
        data.process(1, ['data/test_data/testing_8_s2.txt']),
        data.process(1, ['data/test_data/testing_9_s2.txt']),
    ])

print np.shape(sequence_groups1)
print np.shape(sequence_groups2)

length = 300

sequence_groups1 = data.transform.default_transform(sequence_groups1)
sequence_groups2 = data.transform.default_transform(sequence_groups2)

#sequence_groups = data.transform.pad_extra(sequence_groups, length)
#sequence_groups = data.transform.augment_pad_truncate_intervals(sequence_groups, length, 20)
#
#plt.plot(sequence_groups[0][len(sequence_groups[0])/20+1][:,0])
#plt.show()
#plt.plot(sequence_groups[0][len(sequence_groups[0])/20*2+1][:,0])
#plt.show()
#plt.plot(sequence_groups[0][len(sequence_groups[0])/20*3+1][:,0])
#plt.show()
#
#1/0
#
#print np.shape(sequence_groups)

color_map = ['purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']


def compare_datasets(digit):
    fig = plt.figure(1, figsize=(8, 14))
    fig.clear()
    axes = fig.subplots(7, 2)
    for channel in range(7):
        for sequence in sequence_groups1[digit]:
            axes[channel][0].plot(sequence[:,channel], alpha=0.5)
        for sequence in sequence_groups2[digit]:
            axes[channel][1].plot(sequence[:,channel], alpha=0.5)
#            axes[channel][1].set_xlim(axes[channel][0].get_xlim())
            axes[channel][1].set_ylim(axes[channel][0].get_ylim())

for digit in range(7):
    compare_datasets(digit)
    plt.show()
    
#def plot_channels_for_single_digit(digit=0, show=True):
#    plt.figure()
#    tmp = np.mean(sequence_groups[digit], axis=0)
#    for i in range(np.shape(sequence_groups[0][0])[1]):
#        plt.plot(tmp[:,i], c=color_map[i])
#    title = 'Digit ' + str(digit)
#    plt.title(title)
#    if show: plt.show()

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


