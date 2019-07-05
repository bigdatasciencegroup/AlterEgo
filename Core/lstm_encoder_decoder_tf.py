import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from display_utils import DynamicConsoleTable
import math
import time
import os.path

import data

abs_path = os.path.abspath(os.path.dirname(__file__))

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))
    
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

#length = 3000 #3000 #600 #2800
    
channels = range(0, 8)
surrounding = 250 #250

labels = [64, 148, 125, 66, 181, 106, 191, 94, 79, 170, 144, 55, 152, 101, 128, 92, 142, 116, 81, 85, 140, 134, 127, 90, 175, 149, 83, 69, 123, 196, 141, 185, 119, 178, 164, 98, 103, 78, 104, 158, 162, 67, 169, 153, 108, 168, 68, 54, 95, 109, 89, 86, 167, 189, 157, 182, 176, 135, 172, 145, 61, 163, 173, 62, 52, 154, 56, 177, 160, 115, 105, 194, 188, 96, 112, 124, 166, 143, 150, 139, 60, 84, 82, 174, 88, 133, 161, 199, 77, 73, 117, 59, 180, 147, 155, 195, 137, 198, 159, 114]

train_1 = data.process_scrambled(labels, ['sen200_train_1.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)

labels = [109, 104, 110, 88, 119, 133, 117, 138, 59, 157, 161, 178, 70, 91, 153, 167, 136, 94, 51, 147, 112, 168, 164, 170, 93, 81, 84, 132, 160, 195, 115, 95, 152, 198, 127, 105, 193, 65, 89, 149, 188, 177, 101, 53, 97, 176, 74, 54, 134, 63, 75, 121, 69, 126, 174, 118, 144, 103, 124, 190, 106, 107, 139, 145, 185, 123, 55, 140, 60, 83, 67, 62, 114, 116, 73, 129, 98, 90, 58, 80, 163, 111, 179, 79, 166, 135, 57, 181, 100, 102, 175, 189, 183, 184, 120, 158, 50, 99, 141, 82]

train_2 = data.process_scrambled(labels, ['sen200_train_2.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)

labels = [168, 165, 171, 79, 188, 58, 56, 173, 85, 52, 181, 108, 61, 163, 179, 142, 152, 54, 190, 104, 144, 123, 192, 131, 154, 125, 67, 117, 155, 72, 176, 141, 55, 95, 148, 113, 126, 186, 172, 139, 121, 129, 175, 143, 84, 150, 162, 136, 94, 74, 101, 63, 68, 98, 167, 151, 197, 140, 199, 71, 97, 92, 166, 164, 69, 161, 107, 115, 189, 102, 195, 149, 100, 157, 120, 110, 194, 65, 114, 130, 105, 196, 51, 75, 138, 132, 76, 159, 62, 193, 135, 111, 184, 127, 64, 112, 183, 57, 147, 82]

train_3 = data.process_scrambled(labels, ['sen200_train_3.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([21]), num_classes=200)

labels = [106, 169, 141, 122, 166, 135, 103, 105, 194, 75, 62, 136, 170, 167, 55, 162, 183, 90, 51, 188, 165, 65, 186, 185, 148, 131, 95, 139, 179, 89, 99, 171, 84, 129, 94, 83, 140, 192, 158, 195, 196, 156, 134, 154, 57, 108, 125, 193, 66, 157, 151, 173, 64, 96, 138, 144, 178, 87, 85, 78, 187, 121, 79, 53, 52, 56, 123, 93, 97, 74, 60, 168, 61, 63, 143, 181, 104, 190, 153, 130, 58, 82, 152, 150, 71, 67, 86, 69, 116, 137, 182, 119, 199, 73, 176, 112, 109, 132, 160, 175]

train_4 = data.process_scrambled(labels, ['sen200_train_4.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)

labels = [141, 125, 118, 62, 180, 61, 122, 123, 166, 145, 78, 142, 199, 151, 132, 115, 102, 177, 181, 56, 182, 140, 73, 80, 152, 76, 186, 168, 150, 67, 52, 129, 105, 95, 124, 59, 70, 87, 103, 172, 100, 164, 196, 88, 163, 58, 66, 106, 133, 153, 86, 138, 55, 54, 136, 149, 171, 193, 176, 195, 121, 192, 135, 79, 173, 92, 77, 97, 119, 112, 90, 114, 108, 191, 167, 147, 117, 190, 155, 57, 128, 175, 83, 162, 84, 183, 75, 148, 91, 65, 187, 63, 98, 198, 161, 89, 81, 107, 85, 139]

train_5 = data.process_scrambled(labels, ['sen200_train_5.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([49]), num_classes=200)

labels = [143, 136, 199, 120, 179, 67, 161, 146, 162, 141, 107, 78, 133, 91, 112, 193, 169, 130, 159, 90, 98, 54, 189, 113, 163, 52, 101, 63, 85, 123, 157, 95, 69, 181, 132, 152, 73, 197, 104, 74, 153, 70, 72, 151, 97, 177, 122, 68, 168, 148, 115, 58, 160, 192, 138, 158, 55, 92, 56, 185, 99, 140, 81, 65, 61, 62, 144, 108, 100, 172, 186, 142, 176, 121, 156, 116, 170, 135, 137, 110, 187, 198, 102, 134, 149, 64, 106, 150, 93, 57, 96, 183, 178, 87, 195, 191, 164, 155, 154, 105]

train_6 = data.process_scrambled(labels, ['sen200_train_6.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)

labels = [95, 150, 60, 83, 154, 193, 130, 158, 54, 152, 172, 112, 195, 93, 186, 149, 159, 105, 175, 81, 56, 59, 138, 58, 109, 72, 66, 183, 198, 178, 160, 69, 179, 115, 84, 99, 77, 125, 101, 108, 168, 136, 165, 120, 61, 76, 173, 62, 82, 148, 51, 164, 88, 194, 137, 117, 86, 100, 64, 197, 155, 67, 161, 85, 124, 123, 50, 114, 121, 53, 187, 169, 116, 145, 96, 199, 57, 185, 91, 65, 151, 94, 167, 147, 75, 126, 191, 143, 111, 128, 140, 89, 107, 180, 142, 102, 181, 129, 122, 133]

train_7 = data.process_scrambled(labels, ['sen200_train_7.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)

training_sequence_groups = data.combine([train_1, train_2, train_3, train_4, train_5, train_6, train_7])
#training_sequence_groups = data.combine([train_1])

#print len(training_sequence_groups)
#print map(len, training_sequence_groups)

lens = map(len, data.get_inputs(training_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)


labels = [182, 51, 164, 112, 160, 77, 141, 86, 135, 148, 138, 61, 189, 121, 62, 68, 87, 109, 161, 115, 169, 177, 152, 171, 168]

test1_1 = data.process_scrambled(labels, ['sen200_test1_1.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)


labels = [111, 125, 181, 58, 195, 60, 103, 91, 97, 56, 171, 70, 139, 150, 118, 188, 183, 73, 119, 63, 124, 122, 190, 114, 145]

test1_2 = data.process_scrambled(labels, ['sen200_test1_2.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)


labels = [145, 64, 66, 187, 70, 183, 170, 152, 144, 178, 146, 56, 181, 59, 173, 111, 130, 150, 118, 80, 166, 74, 83, 165, 77]

test1_3 = data.process_scrambled(labels, ['sen200_test1_3.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)


labels = [124, 188, 154, 58, 100, 79, 94, 130, 146, 72, 54, 66, 187, 135, 125, 149, 122, 87, 172, 63, 68, 80, 115, 180, 183]

test1_4 = data.process_scrambled(labels, ['sen200_test1_4.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)


labels = [186, 90, 128, 166, 105, 160, 95, 158, 92, 168, 187, 199, 173, 178, 124, 118, 97, 130, 66, 141, 64, 86, 174, 162, 54]

test1_5 = data.process_scrambled(labels, ['sen200_test1_5.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)


labels = [76, 180, 65, 187, 139, 127, 117, 96, 137, 152, 149, 63, 172, 142, 175, 71, 118, 186, 156, 69, 191, 99, 124, 54, 126]

test1_6 = data.process_scrambled(labels, ['sen200_test1_6.txt'], channels=channels, sample_rate=250,
                                 surrounding=surrounding, exclude=set([]), num_classes=200)


test1_sequence_groups = data.combine([test1_1, test1_2, test1_3, test1_4, test1_5, test1_6])
#test1_sequence_groups = data.combine([test1_1])

#print len(test1_sequence_groups)
#print map(len, test1_sequence_groups)

lens = map(len, data.get_inputs(test1_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)


#labels = [41, 14, 36, 39, 28, 11, 17, 21, 38, 29, 8, 35, 23, 42, 45, 31, 10, 46, 4, 13, 24, 16, 15, 19, 40]
#
#test2_1 = data.process_scrambled(labels, ['sen200_test2_1.txt'], channels=channels, sample_rate=250,
#                                 surrounding=surrounding, exclude=set([]), num_classes=200)
#
#
#labels = [22, 29, 17, 37, 46, 26, 30, 18, 20, 43, 12, 7, 27, 48, 15, 47, 1, 45, 23, 40, 19, 6, 13, 44, 4]
#
#test2_2 = data.process_scrambled(labels, ['sen200_test2_2.txt'], channels=channels, sample_rate=250,
#                                 surrounding=surrounding, exclude=set([]), num_classes=200)
#
#
#labels = [48, 29, 43, 37, 15, 33, 17, 1, 7, 25, 31, 32, 5, 44, 26, 40, 24, 4, 30, 19, 34, 38, 6, 27, 36]
#
#test2_3 = data.process_scrambled(labels, ['sen200_test2_3.txt'], channels=channels, sample_rate=250,
#                                 surrounding=surrounding, exclude=set([]), num_classes=200)
#
#
#labels = [10, 31, 21, 11, 6, 8, 32, 9, 22, 48, 1, 44, 45, 36, 27, 3, 40, 38, 25, 47, 41, 13, 16, 37, 34]
#
#test2_4 = data.process_scrambled(labels, ['sen200_test2_4.txt'], channels=channels, sample_rate=250,
#                                 surrounding=surrounding, exclude=set([]), num_classes=200)
#
#
#labels = [15, 41, 1, 9, 6, 23, 29, 18, 13, 7, 44, 5, 26, 12, 36, 24, 16, 47, 0, 39, 19, 33, 31, 32, 38]
#
#test2_5 = data.process_scrambled(labels, ['sen200_test2_5.txt'], channels=channels, sample_rate=250,
#                                 surrounding=surrounding, exclude=set([]), num_classes=200)
#
#
#labels = [3, 41, 39, 8, 22, 40, 1, 5, 30, 10, 23, 11, 15, 49, 20, 32, 45, 47, 19, 7, 36, 24, 42, 29, 43]
#
#test2_6 = data.process_scrambled(labels, ['sen200_test2_6.txt'], channels=channels, sample_rate=250,
#                                 surrounding=surrounding, exclude=set([]), num_classes=200)
#
#
#test2_sequence_groups = data.combine([test2_1, test2_2, test2_3, test2_4, test2_5, test2_6])
#
##print len(test2_sequence_groups)
##print map(len, test2_sequence_groups)
#
#lens = map(len, data.get_inputs(test2_sequence_groups)[0])
#print min(lens), np.mean(lens), max(lens)






#training_sequence_groups = data.combine([training_sequence_groups, test1_sequence_groups])

validation_sequence_groups = test1_sequence_groups
#validation_sequence_groups = test2_sequence_groups


# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)


train_sequences = transform_data(train_sequences)
val_sequences = transform_data(val_sequences)



#train_sequences = train_sequences[:20]
#train_labels = train_labels[:20]



# Calculate sample weights
#class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
#train_weights = class_weights[list(train_labels)]
train_weights = np.ones(len(train_labels))

#train_labels = tf.keras.utils.to_categorical(train_labels)
#val_labels = tf.keras.utils.to_categorical(val_labels)

words = np.array(['i', 'am', 'you', 'are', 'the', 'want', 'need', 'cold', 'hot', 'food', 'where', 'what', 'how', 'feeling', 'doing', 'tired', 'water', 'hungry', 'thirsty', 'hello'])

label_map = [[11, 1, 0, 14], [10, 4, 9], [3, 10, 2], [12, 3, 2, 5, 0], [11, 3, 2, 14, 4], [14, 2, 12], [10, 16, 3, 14, 11, 0], [1, 11, 0, 4, 16, 10], [13, 4, 6, 9], [1, 7, 16], [8, 16, 12, 9, 7], [12, 1, 5, 8, 9, 7], [19, 18, 17, 5, 4, 15], [6, 0, 3, 11, 2, 17], [19, 18, 15, 13, 4, 6], [12, 11, 10, 2, 3, 13], [0, 1, 14], [7, 16, 10], [13, 12, 7, 16], [1, 13, 15], [18, 17, 3, 10, 8, 19], [12, 2, 14, 4, 17], [16, 1, 15, 14, 11, 0], [8, 6, 3, 18, 19, 5], [1, 0, 4, 16, 2, 6], [13, 15, 12, 8, 5, 9], [14, 11, 6, 9], [7, 13, 15], [9, 7, 13, 8], [5, 2, 18, 19, 17], [1, 0, 3, 4, 12], [15, 16, 5, 11, 10, 9], [7, 9, 13, 18], [19, 1, 2, 17], [14, 11, 0, 5], [3, 10, 4, 13, 12], [16, 10, 8, 9], [6, 14, 4, 17, 19, 18], [1, 10, 0], [3, 11, 2], [15, 19, 8, 16, 12, 18], [17, 0, 6, 2], [9, 3, 16, 10, 4, 14], [6, 11, 1, 15, 7, 13], [13, 7, 9], [6, 14, 15, 12, 8], [5, 6, 18], [10, 17, 19, 14, 11, 4], [12, 2, 3, 13], [0, 1, 17], [17, 8, 9], [5, 6, 16], [19, 3, 18], [14, 4, 7], [12, 0, 1, 10, 16], [5, 11, 2, 6], [6, 9, 7, 13], [5, 18, 15, 17], [8, 4, 13, 19], [11, 0, 1, 14], [10, 3, 2, 7, 16], [7, 9, 12, 6], [19, 15, 18, 8, 2, 6], [4, 16, 10, 0, 3, 14], [13, 11, 1, 15], [18, 15, 7, 9], [17, 8, 12, 15], [19, 10, 0, 5], [12, 2, 3, 13, 15, 4], [16, 15, 14, 11, 6], [5, 9, 7, 1, 17], [8, 19, 3, 18], [4, 9, 2, 0, 5, 11], [13, 7, 16], [10, 1, 14], [18, 12, 8, 17], [19, 3, 14, 4], [5, 11, 0], [16, 10, 2, 5], [1, 13, 17, 5, 12], [3, 18, 17, 8, 19], [8, 0, 1, 14, 11], [10, 2, 5, 4], [19, 18, 2, 3, 11, 6], [12, 1, 0, 4, 8], [13, 7, 16, 15, 9], [12, 7, 16], [7, 13, 17], [1, 19, 10, 3, 2, 18], [0, 5, 4, 16, 10], [6, 14, 11], [7, 13, 12, 8], [7, 9, 18, 15], [19, 11, 4, 17], [14, 3, 17], [0, 1, 2], [10, 8, 16], [18, 6, 14, 13, 12], [19, 15, 2, 10, 16], [11, 3, 4, 0, 13, 15], [12, 1, 14], [18, 8, 7, 9], [7, 9, 8, 17], [19, 17, 0, 5, 2, 6], [3, 10, 4, 16], [12, 1, 14, 11], [7, 9, 15, 13, 6, 8], [8, 9, 18, 5], [10, 0, 15, 4, 12, 19], [6, 3, 11, 2, 17], [8, 9, 16, 1, 14], [19, 5, 3, 18], [0, 2, 16, 10, 4, 7], [6, 11, 1, 15], [5, 7, 13, 17], [12, 8, 9], [3, 18, 19], [10, 2, 14], [11, 0, 5, 4], [19, 18, 7, 16, 15], [6, 3, 17], [1, 10, 0, 5], [12, 4, 9], [13, 7, 9, 2, 12, 16], [17, 18, 14, 11, 6], [10, 2, 3, 19, 15], [1, 11, 0, 5], [14, 4, 6], [6, 15, 13, 7, 16, 1], [12, 15, 13, 17, 14], [18, 13, 2, 8, 16, 19], [0, 5, 4], [14, 11, 3, 12, 15], [5, 7, 13, 15, 19], [8, 5, 4, 3, 17, 18], [6, 2, 0, 1, 15], [12, 1, 13, 11, 16], [10, 8, 9], [14, 3, 17, 18, 11, 19], [10, 4, 16, 0, 1], [19, 15, 14, 12, 2, 5], [13, 17, 10, 9, 18], [15, 0, 6, 4, 8, 16], [10, 11, 3, 2, 17], [15, 12, 7, 1, 14], [17, 7, 9, 8], [3, 14, 19, 6, 4, 18], [12, 1, 0, 6, 16], [2, 5, 11, 18, 13, 10], [8, 7, 13, 15], [19, 14, 3, 9], [0, 1, 4], [6, 2, 11], [15, 7, 9, 13, 12, 17], [6, 9, 10, 8, 16], [0, 5, 3, 18, 19], [14, 11, 2], [13, 4, 7, 16], [5, 17, 12, 1, 15, 10], [19, 3, 4, 18, 8, 16], [2, 5, 11], [10, 1, 15, 0, 5], [7, 13, 12], [17, 18, 7, 9, 6], [19, 0, 4, 3, 2, 8], [11, 13, 12, 5, 16, 10], [1, 7, 9], [18, 6, 9, 8, 5], [19, 3, 17], [1, 14, 4], [12, 15, 0, 2], [6, 14, 7, 13, 19], [10, 11, 6, 9], [6, 16, 5, 9, 8], [11, 2, 5, 18, 6, 17], [12, 4, 3, 10, 1, 0], [18, 8, 9], [19, 17, 2, 10, 0], [11, 3, 4, 1], [15, 9, 8, 16, 13, 17], [12, 8, 16], [19, 6, 16, 0, 18], [12, 11, 2, 3, 15, 4], [10, 1, 13, 15], [15, 8, 9, 17, 14], [7, 13, 19], [11, 10, 1, 0, 2, 18], [9, 3, 17], [16, 4, 13, 12, 7], [15, 6, 7, 9], [14, 19, 2, 18, 5, 16], [1, 17, 0, 3, 4, 7], [14, 11, 6], [12, 7, 9], [19, 10, 8, 6, 14], [2, 5, 3, 18], [1, 0, 14, 4, 12, 17], [10, 11, 19, 6, 9, 18], [5, 3, 8], [11, 10, 0, 6, 2, 7]]

num_classes = 20

label_map = map(lambda label_seq: tf.keras.utils.to_categorical(label_seq, num_classes=num_classes), label_map)

train_labels = np.array(map(lambda i: label_map[i], train_labels))
val_labels = np.array(map(lambda i: label_map[i], val_labels))

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)



learning_rate = 1e-4 #5e-4 #1e-3
#dropout_rate = 0.4

sample_rate = 250
num_channels = len(channels)





#inputs = tf.placeholder(tf.float32,[None, None, num_channels]) #[batch_size,timestep,features]
#targets = tf.sparse_placeholder(tf.int32)
#sequence_lengths = tf.placeholder(tf.int32, [None])
#weights = tf.placeholder(tf.float32, [None])
#training = tf.placeholder(tf.bool)
#batch_size = tf.shape(inputs)[0]
#max_timesteps = tf.shape(inputs)[1]



batch_size = 1 #50


Y_VOCAB_SIZE = 20
Y_MAX_LENGTH = 6


state_size = 128 #128
encoder_layers = 3 #3
decoder_layers = 3 #3

encoder_x = tf.placeholder(tf.float32, shape=[None, None, num_channels]) #[batch_size, X_MAX_LENGTH]
decoder_x = tf.placeholder(tf.float32, shape=[None, None, Y_VOCAB_SIZE]) #[batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE]
y = tf.placeholder(tf.float32, shape=[None, None, Y_VOCAB_SIZE])#[batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE]
init_state = tf.placeholder(tf.float32, [encoder_layers, 2, batch_size, state_size])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)

########
#
# METHOD: lstm_model()
# DESCRIPTION: Create TensorFlow LSTM Model to process our data in the Encoder/Decoder Architecture:
#              NOTE: Attention is not yet implemented!!
# PARAMS:
#           data: x TensorFlow placeholder data.
# RETURNS:
#           decoder_final_state: Final state for Decoder Network.
#           logits: Un-normalised logits to pass to loss function.
#           labels: y TensorFlow placeholder data.
#           prediction: Softmax prediction output for translation.
#
########
def lstm_model():

    with tf.device('/cpu:0'):

#        with tf.variable_scope('encoder_word_embeddings'):
#
#            word_embeddings = tf.get_variable('encoder_word_embeddings', [X_VOCAB_SIZE, nn_config.embedding_size])
#            encoder_embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, encoder_x)
#            encoder_embedded_word_ids = tf.reshape(encoder_embedded_word_ids, [-1, X_MAX_LENGTH, nn_config.embedding_size])

        with tf.variable_scope('encoder'):
            ####
            #
            # LSTM ENCODER
            #
            ####
            # Forward passes
            state_per_layer_list = tf.unstack(init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                for idx in range(encoder_layers)])

            encoder_stacked_cell = []

            for _ in range(encoder_layers):
                encoder_single_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
                if _ == 0:
                    with tf.name_scope('encoder_dropout') as scope:
                        encoder_single_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_single_cell,
                                                        output_keep_prob=0.75)  # add dropout to first LSTM layer only.
                encoder_stacked_cell.append(encoder_single_cell)

            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_stacked_cell, state_is_tuple=True)
            
            print encoder_cell
            print encoder_x
            print rnn_tuple_state

            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                             inputs=encoder_x,
                                                             initial_state=rnn_tuple_state)
            
            print encoder_outputs

            del encoder_outputs, encoder_cell, state_per_layer_list, encoder_stacked_cell

#        with tf.variable_scope('decoder_word_embeddings'):
#
#            word_embeddings = tf.get_variable('decoder_word_embeddings', [Y_VOCAB_SIZE, nn_config.embedding_size])
#            decoder_embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, decoder_x)
#            decoder_embedded_word_ids = tf.reshape(decoder_embedded_word_ids,
#                                                        [-1, Y_MAX_LENGTH, Y_VOCAB_SIZE * nn_config.embedding_size])

        with tf.variable_scope('decoder'):
            ####
            #
            # LSTM DECODER
            #
            ####
            decoder_stacked_cell = []
            for _ in range(decoder_layers):
                decoder_single_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
                if _ == 0:
                    with tf.name_scope('decoder_dropout') as scope:
                        decoder_single_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_single_cell,
                                                    output_keep_prob=0.75)  # add dropout to first LSTM layer only.
                decoder_stacked_cell.append(decoder_single_cell)

            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_stacked_cell, state_is_tuple=True)

            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                             inputs=decoder_x,
                                                             initial_state=encoder_final_state)

            outputs = tf.reshape(decoder_outputs, [-1, state_size])

            with tf.name_scope('decoder_hidden_states') as scope:
                W2 = tf.Variable(tf.random_normal([state_size, Y_VOCAB_SIZE]), dtype=tf.float32)
                b2 = tf.Variable(tf.zeros([1, Y_VOCAB_SIZE]), dtype=tf.float32)

                logits = tf.matmul(outputs, W2) + b2 # Broadcasted addition
                prediction = tf.nn.softmax(logits)
            labels = y

        return decoder_final_state, labels, logits, prediction


    
current_state, labels, logits, prediction = lstm_model()

with tf.name_scope('cross_entropy') as scope:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(tf.multiply(loss, weights))

with tf.name_scope('optimizer') as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    
#logits_sequence = tf.argmax(logits, axis=1)
#labels_sequence = tf.argmax(labels, axis=1)
#
#idx = tf.where(tf.not_equal(logits_sequence, 0)) # MUST BE FIXED
#logits_sparse = tf.SparseTensor(idx, tf.gather_nd(logits_sequence, idx), tf.cast(tf.shape(logits_sequence), tf.int64))
#
#idx = tf.where(tf.not_equal(labels_sequence, 0)) # MUST BE FIXED
#labels_sparse = tf.SparseTensor(idx, tf.gather_nd(labels_sequence, idx), tf.cast(tf.shape(labels_sequence), tf.int64))
#    
#error = tf.reduce_mean(tf.edit_distance(logits_sparse, labels_sparse, normalize=True))


print current_state
print labels
print logits
print prediction
    
    
    


num_epochs = 20000 #200

num_training_samples = len(train_sequences)
num_validation_samples = len(val_sequences)
num_training_batches = max(1, int(num_training_samples / batch_size))
num_validation_batches = max(1, int(num_validation_samples / batch_size))
start_time = None
last_time = None

# Table display
progress_bar_size = 20
max_batches = max(num_training_batches, num_validation_batches)
layout = [
    dict(name='Ep.', width=len(str(num_epochs)), align='center'),
    dict(name='Batch', width=2*len(str(max_batches))+1, align='center'),
#    dict(name='', width=0, align='center'),
    dict(name='Progress/Timestamp', width=progress_bar_size+2, align='center'),
    dict(name='ETA/Elapsed', width=7, suffix='s', align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Train Loss', width=8, align='center'),
    dict(name='Train Err', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Val Loss', width=8, align='center'),
    dict(name='Val Err', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Min Val Err', width=7, align='center'),
]

training_losses = []
training_errors = []
validation_losses = []
validation_errors = []

since_training = 0
def update_table(epoch, batch, training_loss, training_error, min_validation_error,
                 validation_loss=None, validation_error=None, finished=False):
    global last_time
    global since_training
    num_batches = num_training_batches if validation_loss is None else num_validation_batches
    progress = int(math.ceil(progress_bar_size * float(batch) / num_batches))
#    progress_string = '[' + '#' * progress + ' ' * (progress_bar_size - progress) + ']'
    status = ' Training' if validation_loss is None else ' Validating'
    status = status[:max(0, progress_bar_size - progress)]
    progress_string = '[' + '#' * progress + status + ' ' * (progress_bar_size - progress - len(status)) + ']'
    now = time.time()
    start_elapsed = now - start_time
    if validation_loss is None:
        epoch_elapsed = now - last_time
        since_training = now
    else:
        epoch_elapsed = now - since_training
    batch_time_estimate = epoch_elapsed / batch if batch else 0.0
    eta_string = batch_time_estimate * (num_batches - batch) or '--'
    if finished:
        epoch_elapsed = now - last_time
        last_time = now
        progress_string = time.strftime("%I:%M:%S %p",time.localtime())+'; '+str(int(start_elapsed*10)/10.)+'s'
        eta_string = epoch_elapsed
        training_losses.append(training_loss)
        training_errors.append(training_error)
        validation_losses.append(validation_loss)
        validation_errors.append(validation_error)
    table.update(epoch + 1, str(batch + 1) + '/' + str(num_batches),
                 progress_string, eta_string, '',
                 training_loss or '--', training_error or '--', '',
                 validation_loss or '--', validation_error or '--', '',
                 min_validation_error if finished else '--')
    
def levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

saver = tf.train.Saver()
with tf.Session() as session:
    tf.global_variables_initializer().run()
    
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    start_time = time.time()
    last_time = start_time
    
    _current_state = np.zeros((encoder_layers, 2, batch_size, state_size))
    
    min_validation_error = float('inf')
    for epoch in range(num_epochs):
        training_loss = 0.0
        training_error = 0.0
        permutation = np.random.permutation(num_training_samples)
        train_sequences = train_sequences[permutation]
        train_labels = train_labels[permutation]
        train_weights = train_weights[permutation]
        training_output = []
        training_edit_distances = []
        for batch in range(num_training_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_training_batches - 1:
                indices = range(batch * batch_size, num_training_samples)
            batch_sequences = train_sequences[indices]
            tmp_lengths = map(len, batch_sequences)
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(tmp_lengths), position=0.0, value=-1e8)
            batch_labels = map(np.array, train_labels[indices]) # CAUSING ISSUE WITH setting an array element with a sequence WHEN batch_size > 1
            batch_weights = train_weights[indices]
            
            update_table(epoch, batch, training_loss / (batch_size * max(1, batch)),
#                         training_error / (batch_size * max(1, batch)), min_validation_error)
                         training_error, min_validation_error)
                        
            training_feed = {encoder_x: batch_sequences, decoder_x: batch_labels, y: batch_labels, init_state: _current_state,
                             weights: batch_weights, training: True}
#            batch_loss, _, _current_state, batch_output = session.run([loss, optimizer, current_state, prediction], training_feed)
            batch_loss, _, __, batch_output = session.run([loss, optimizer, current_state, prediction], training_feed)
            
        # FIX BATCH OUTPUT BEING ONLY TWO DIMENSIONAL, not including batch size, then change below
            
            training_loss += batch_loss * len(indices)
#            training_error += batch_error * len(indices)

            training_output.append(batch_output)
    
            # ONLY WORKS WITH BATCH SIZE 1
            target_seq = np.argmax(batch_labels[0], axis=1)
            training_edit_distances.append(float(levenshtein(target_seq, np.argmax(batch_output, axis=1)))/len(target_seq))
            training_error = np.mean(training_edit_distances)
            
        training_loss /= num_training_samples
#        training_error /= num_training_samples
        
        train_target_seqs = map(lambda x: list(np.argmax(x, axis=1)), train_labels)
        train_predicted_seqs = map(lambda x: list(np.argmax(x, axis=1)), training_output)
        
        training_edit_distances = []
        for i in range(len(train_target_seqs)):
            training_edit_distances.append(float(levenshtein(train_target_seqs[i], train_predicted_seqs[i]))/len(train_target_seqs[i]))
        training_error = np.mean(training_edit_distances)
                
        validation_loss = 0.0
        validation_error = 0.0
        permutation = np.random.permutation(num_validation_samples)
        val_sequences = val_sequences[permutation]
        val_labels = val_labels[permutation]
        validation_output = []
        validation_edit_distances = []
        for batch in range(num_validation_batches):         
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_validation_batches - 1:
                indices = range(batch * batch_size, num_validation_samples)
            batch_sequences = val_sequences[indices]
            tmp_lengths = map(len, batch_sequences)
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(tmp_lengths), position=0.0, value=-1e8)
            batch_labels = map(np.array, val_labels[indices])
            batch_weights = np.ones(len(indices))
                        
            update_table(epoch, batch, training_loss, training_error, min_validation_error,
                         validation_loss / (batch_size * max(1, batch)),
#                         validation_error / (batch_size * max(1, batch)))
                         validation_error)
                                    
            validation_feed = {encoder_x: batch_sequences, decoder_x: batch_labels, y: batch_labels, init_state: _current_state,
                               weights: batch_weights, training: False}
#            batch_loss, _current_state, batch_output = session.run([loss, current_state, prediction], validation_feed)
            batch_loss, __, batch_output = session.run([loss, current_state, prediction], validation_feed)
                
            validation_loss += batch_loss * len(indices)
#            validation_error += batch_error * len(indices)

            validation_output.append(batch_output)
    
            # ONLY WORKS WITH BATCH SIZE 1
            target_seq = np.argmax(batch_labels[0], axis=1)
            validation_edit_distances.append(float(levenshtein(target_seq, np.argmax(batch_output, axis=1)))/len(target_seq))
            validation_error = np.mean(validation_edit_distances)
                
        validation_loss /= num_validation_samples
#        validation_error /= num_validation_samples

        val_target_seqs = map(lambda x: list(np.argmax(x, axis=1)), val_labels)
        val_predicted_seqs = map(lambda x: list(np.argmax(x, axis=1)), validation_output)
        
        validation_edit_distances = []
        for i in range(len(val_target_seqs)):
            validation_edit_distances.append(float(levenshtein(val_target_seqs[i], val_predicted_seqs[i]))/len(val_target_seqs[i]))
        validation_error = np.mean(validation_edit_distances)
        
        min_validation_error = min(validation_error, min_validation_error)
                                    
        update_table(epoch, batch, training_loss, training_error,
                     min_validation_error, validation_loss, validation_error, finished=True)
        
        print
        print
        print 'Training:'
        for i in range(len(train_target_seqs[:10])):
            print train_target_seqs[i], train_predicted_seqs[i]
        print
        print 'Validation:'
        for i in range(len(val_target_seqs[:10])):
            print val_target_seqs[i], val_predicted_seqs[i]
        
        reprint_header = (epoch+1) % 10 == 0 and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()
