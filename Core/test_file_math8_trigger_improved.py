import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys

import data

test_model = True
#channels = range(1, 8) # Must be same as trained model if test_model==True
#channels = range(4, 8) # Must be same as trained model if test_model==True
channels = eval(sys.argv[1])
print channels
#channels = [1, 3, 4] # DO NOT CHANGE

#word_map = ['hello', 'assistance', 'thank you']
#word_map = ['hello', 'elephant', 'reboot computer']
#word_map = ['left', 'right', 'rotate', 'silence']
word_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'plus', 'minus', 'times', 'divided by', 'undo', 'silence']

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


num_classes = 15 #len(filter(lambda x: '.txt' in x, os.listdir('patient_data')))
length = 450 #600 DO NOT CHANGE

####################
#### Model (MUST BE SAME AS patient_train.py)
learning_rate = 1e-4
dropout_rate = 0.4

inputs = tf.placeholder(tf.float32,[None, length, len(channels)]) #[batch_size,timestep,features]
targets = tf.placeholder(tf.int32, [None, num_classes])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)

conv1 = tf.layers.conv1d(inputs, 400, 12, activation=tf.nn.relu, padding='valid')
conv1 = tf.layers.max_pooling1d(conv1, 2, strides=2)
conv2 = tf.layers.conv1d(conv1, 400, 6, activation=tf.nn.relu, padding='valid')
conv2 = tf.layers.max_pooling1d(conv2, 2, strides=2)
conv3 = tf.layers.conv1d(conv2, 400, 3, activation=tf.nn.relu, padding='valid')
conv3 = tf.layers.max_pooling1d(conv3, 2, strides=2)
conv4 = tf.layers.conv1d(conv3, 400, 3, activation=tf.nn.relu, padding='valid')
conv4 = tf.layers.max_pooling1d(conv4, 2, strides=2)
conv5 = tf.layers.conv1d(conv4, 400, 3, activation=tf.nn.relu, padding='valid')
conv5 = tf.layers.max_pooling1d(conv5, 2, strides=2)
dropout = tf.layers.dropout(conv5, dropout_rate, training=training)
reshaped = tf.reshape(dropout, [-1, np.prod(dropout.shape[1:])])
fc1 = tf.layers.dense(reshaped, 250, activation=tf.nn.relu)
logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax)

loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
####################
        
predictions = []
saver = tf.train.Saver()
with tf.Session() as session:
    if test_model:
        tf.global_variables_initializer().run()
        saver.restore(session, 'math8_trigger_model' + sys.argv[2] + '.ckpt')
        
    displayed = 0
    step = 1
    bar_size = 20
    last_test_output = None
    last_input_indices = None
    selected = None
    ground_truth_count = -1
    ground_truths = []
    def on_data(history, trigger_history, index_history, count, samples_per_update):
        global displayed
        global last_test_output
        global last_input_indices
        global selected
        global predictions
        global ground_truth_count
        global ground_truths
        if count - displayed > step:
            
            labels = [11, 6, 7, 14, 9, 5, 3, 10, 6, 12, 12, 12, 0, 11, 14, 7, 13, 12, 8, 7, 14, 2, 10, 13, 9, 12, 4, 5, 11, 14, 4, 5, 7, 8, 1, 6, 4, 10, 4, 2, 14, 3, 4, 13, 4, 8, 11, 10, 5, 6, 8, 11, 2, 4, 9, 13, 0, 10, 1, 12, 5, 6, 5, 13, 6, 3, 5, 3, 11, 9, 12, 6, 2, 8, 0, 14, 3, 14, 5, 11, 1, 7, 0, 13, 7, 2, 7, 7, 0, 14, 1, 10, 11, 10, 8, 13, 1, 3, 3, 1, 10, 2, 12, 2, 6, 3, 0, 8, 9, 0, 2, 8, 9, 0, 9, 1, 9, 1, 13, 4]
            
            if test_model:
                start = None
                end = None
                trigger = None
                for i in range(len(trigger_history))[::-1]:
                    if trigger_history[i] == 1 and trigger == False:
                        end = i
                        trigger = True
                    if trigger_history[i] == 0:
                        if trigger:
                            start = i
                            break
                        trigger = False
                if start and end:
                    signal_length = end - start
                    side_padding = (length - signal_length) // 2
                    print signal_length
                    print end - start > 50
                    print end + side_padding < len(history)
                    print start - side_padding - 1 >= 0
                    if end - start > 50 and end + side_padding + 1 < len(history) and start - side_padding - 1 >= 0:
    #                    input_indices = index_history[start:end+1]
                        start_modified, end_modified = start-(length-signal_length-side_padding-1), end+side_padding
                        input_indices = index_history[start_modified:end_modified+1]
                        print 'DEBUG'
                        if not last_input_indices is None:
                            print len(input_indices) != len(last_input_indices)
                            if len(input_indices) == len(last_input_indices):
                                print not np.all(input_indices == last_input_indices)
                        if last_input_indices is None or (len(input_indices) != len(last_input_indices) 
                            or not np.all(input_indices == last_input_indices)):
                            last_input_indices = input_indices
                            
                            ground_truth_count += 1

                            print start, end
                            print start_modified, end_modified

                            sequence = history[start_modified:end_modified+1,channels]
                            test_feed = {inputs: [sequence], training: False}
                            last_test_output = session.run(logits, test_feed)[0]
                            selected = np.argmax(last_test_output)
#                            os.system('say ' + str(word_map[selected]) + ' &')
                            predictions.append(selected)
                            
                            ground_truths.append(labels[ground_truth_count])
            
            print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i+1) for i in range(8)])
            print str('{:.1f}'.format(count/250.)) + 's\t\t' + '\t'.join(
                map(lambda (i, x): '{:f}'.format(x) if i in channels else '--\t', enumerate(history[-1])))
            print
            if test_model and last_test_output is not None:
                for i in range(len(last_test_output)):
                    n = int(last_test_output[i] * bar_size)
                    print i, '[' + '#' * n + ' ' * (bar_size - n) + ']', last_test_output[i]
                print word_map[selected]
                print
            print 'Truth\t\t', ground_truths
            print 'Predicted\t', predictions

            
        displayed = count // step * step
        
    data.from_file.start('math8_test.txt',
                      on_data, channels=channels, transform_fn=transform_data,
                      history_size=1500, shown_size=1200, override_step=65, sample_rate=250, speed=10.0)
    
#    data.serial.start('/dev/tty.usbserial-DQ007UBV',
#                      on_data, channels=channels, transform_fn=transform_data,
#                      history_size=1500, shown_size=1200, override_step=75) #95