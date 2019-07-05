import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import time
import sys

import data

test_model = True
#channels = range(1, 8) # Must be same as trained model if test_model==True
#channels = range(4, 8)
channels = range(0, 3)
#channels = eval(sys.argv[1])
#print channels
#channels = [1, 3, 4] # DO NOT CHANGE

#word_map = ['hello', 'assistance', 'thank you']
#word_map = ['hello', 'elephant', 'reboot computer']
#word_map = ['left', 'right', 'rotate', 'silence']
word_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'plus', 'yes yes yes', 'silence']

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


num_classes = 12 #len(filter(lambda x: '.txt' in x, os.listdir('patient_data')))
num_classes_act = 2 #len(filter(lambda x: '.txt' in x, os.listdir('patient_data')))
length = 450 #600 DO NOT CHANGE
length_act = 150 #600 DO NOT CHANGE

####################
graph1 = tf.Graph()
with graph1.as_default():
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
    
    saver1 = tf.train.Saver()
####################

####################
graph2 = tf.Graph()
with graph2.as_default():
    #### Model (MUST BE SAME AS patient_train.py)
    learning_rate = 1e-4
    dropout_rate = 0.4

    inputs_act = tf.placeholder(tf.float32,[None, length_act, len(channels)]) #[batch_size,timestep,features]
    targets_act = tf.placeholder(tf.int32, [None, num_classes_act])
    weights_act = tf.placeholder(tf.float32, [None])
    training_act = tf.placeholder(tf.bool)

    conv1_act = tf.layers.conv1d(inputs_act, 400, 12, activation=tf.nn.relu, padding='valid')
    conv1_act = tf.layers.max_pooling1d(conv1_act, 2, strides=2)
    conv2_act = tf.layers.conv1d(conv1_act, 400, 6, activation=tf.nn.relu, padding='valid')
    conv2_act = tf.layers.max_pooling1d(conv2_act, 2, strides=2)
    conv3_act = tf.layers.conv1d(conv2_act, 400, 3, activation=tf.nn.relu, padding='valid')
    conv3_act = tf.layers.max_pooling1d(conv3_act, 2, strides=2)
    conv4_act = tf.layers.conv1d(conv3_act, 400, 3, activation=tf.nn.relu, padding='valid')
    conv4_act = tf.layers.max_pooling1d(conv4_act, 2, strides=2)
    dropout_act = tf.layers.dropout(conv4_act, dropout_rate, training=training_act)
    reshaped_act = tf.reshape(dropout_act, [-1, np.prod(dropout_act.shape[1:])])
    fc1_act = tf.layers.dense(reshaped_act, 250, activation=tf.nn.relu)
    logits_act = tf.layers.dense(fc1_act, num_classes_act, activation=tf.nn.softmax)

    loss_act = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits_act, labels=targets_act), weights_act))

    optimizer_act = tf.train.AdamOptimizer(learning_rate).minimize(loss_act)

    correct_act = tf.equal(tf.argmax(logits_act,1), tf.argmax(targets_act,1))
    accuracy_act = tf.reduce_mean(tf.cast(correct_act, tf.float32))
    
    saver2 = tf.train.Saver()
####################
        
predictions = []
certainty_sums = [0.0] * num_classes
activated_timestamp = 0
produced_timestamp = 0
last_activated = False
last_activated_start = -1
last_test_output = None
update_count = 0
last_activated_count = 0
with tf.Session(graph=graph1) as session1:
    if test_model:
        tf.global_variables_initializer().run()
        saver1.restore(session1, 'math23_trigger_model.ckpt')
    with tf.Session(graph=graph2) as session2:
        if test_model:
            tf.global_variables_initializer().run()
            saver2.restore(session2, 'math23_activation_small_model.ckpt')
        
        displayed = 0
        step = 1
        bar_size = 20
        last_test_output = None
        last_input_indices = None
        selected = None
        ground_truth_count = -1
        ground_truths = []
        trigger_predictions = []
        def on_data(history, trigger_history, index_history, count, samples_per_update):
            global displayed
            global last_test_output
            global last_input_indices
            global selected
            global predictions
            global certainty_sums
            global activated_timestamp
            global produced_timestamp
            global ground_truth_count
            global ground_truths
            global trigger_predictions
            global last_activated
            global last_activated_start
            global last_test_output
            global update_count
            global last_activated_count
            if count - displayed > step:

                labels = [5, 3, 5, 7, 9, 9, 8, 5, 10, 10, 7, 3, 11, 9, 4, 7, 4, 1, 0, 4, 6, 7, 2, 11, 11, 4, 4, 6, 10, 8, 6, 4, 5, 6, 2, 4, 10, 8, 11, 1, 11, 5, 10, 3, 11, 5, 8, 0, 6, 6, 2, 10, 0, 8, 3, 9, 5, 5, 6, 11, 3, 1, 2, 7, 0, 10, 7, 2, 7, 7, 0, 11, 1, 8, 10, 1, 3, 3, 1, 2, 2, 6, 3, 0, 8, 9, 0, 2, 8, 9, 0, 9, 1, 9, 1, 4]
#                trigger_labels = [11, 6, 8, 14, 9, 5, 3, 10, 6, 5, 13, 11, 0, 11, 9, 0, 13, 11, 8, 7, 9, 2, 10, 13, 0, 12, 4, 13, 10, 14, 10, 5, 7, 8, 4, 6, 4, 10, 4, 2, 14, 3, 4, 13, 13, 8, 11, 10, 4, 6, 8, 11, 6, 4, 13, 13, 0, 10, 8, 12, 11, 6, 4, 13, 6, 9, 4, 3, 12, 5, 11, 8, 9, 8, 0, 14, 3, 8, 4, 11, 4, 7, 0, 13, 7, 5, 0, 3, 0, 14, 14, 10, 8, 10, 8, 13, 0, 3, 3, 4, 10, 6, 11, 2, 6, 3, 3, 8, 2, 0, 2, 9, 9, 0, 3, 2, 13, 2, 10, 5]

                if test_model:
                    update_count += 1
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
    #                    print signal_length
    #                    print end - start > 50
    #                    print end + side_padding < len(history)
    #                    print start - side_padding - 1 >= 0
                        if end - start > 50 and end + side_padding + 1 < len(history) and start - side_padding - 1 >= 0:
        #                    input_indices = index_history[start:end+1]
                            start_modified, end_modified = start-(length-signal_length-side_padding-1), end+side_padding
                            input_indices = index_history[start_modified:end_modified+1]
                            if last_input_indices is None or (len(input_indices) != len(last_input_indices) 
                                or not np.all(input_indices == last_input_indices)):
                                last_input_indices = input_indices
                            
                                ground_truth_count += 1
    
#                                print start, end
#                                print start_modified, end_modified
    
                                sequence = history[start_modified:end_modified+1,channels]
                                test_feed = {inputs: [sequence], training: False}
                                last_test_output = session1.run(logits, test_feed)[0]
                                selected = np.argmax(last_test_output)
#                                os.system('say ' + str(word_map[selected]) + ' &')
    
                                ground_truths.append(labels[ground_truth_count])
#                                trigger_predictions.append(trigger_labels[ground_truth_count])
                                trigger_predictions.append(selected)
                                predictions.append(None)
#                                os.system('say -r 400 ' + str(word_map[labels[ground_truth_count+1]]) + ' &')

                if test_model:
                    sequence_act = history[-length*2:,channels]
#                    sequence_act = sequence
                    activation_feed = {inputs_act: [sequence_act[i:i+length_act] \
                                                    for i in range(0, len(sequence_act), length_act//8) if len(sequence_act)-i >= length_act],
                                       training_act: False}
                    activation_output = session2.run(logits_act, activation_feed)
#                    selected = np.argmax(test_output)
    #                certainty_sums += test_output
#                    certainty_sums += np.square(test_output)
        
        
#                    activated = np.all(1-np.argmax(activation_output, axis=1) == 1)

                    activation_windows = np.array(map(lambda x: x[0] > 0.98, activation_output)) # 0.99999
#                    start = (len(activation_windows)-8)//2
#                    activated = sum(1-activation_windows[:start])>=1 and np.all(activation_windows[start:start+len(activation_windows)])
#                    activated = sum(activation_windows) >= len(activation_windows) - 2

                    activated_start, activated_end = None, None
                    max_length = 0
                    tmp_start = -1
                    for i in range(len(activation_windows)):
                        if tmp_start > -1 \
                            and i >= 5 and np.all(activation_windows[i-5:i-4]) and not np.any(activation_windows[i-4:i]): # 4
                            if 8 <= (i-4) - tmp_start and (i-4) < len(activation_windows) \
                                and np.sum(activation_windows[tmp_start:i-4]) > 0.75 * ((i-4) - tmp_start):
                                activated_start, activated_end = tmp_start, i-4
                                break
                        if i >= 4 and not np.any(activation_windows[i-4:i]) and activation_windows[i]:
                            tmp_start = i
                            
                    if activated_start and activated_end:
                        print activated_start, activated_end, activated_end - activated_start
                        
                    activated = activated_start and activated_end and (activated_start - last_activated_start >= 2 
                                                                       or update_count - last_activated_count > 10)
                    
#                    activated = activated_start and activated_end and (update_count - last_activated_count > 10)
        
#                    activated = activated_start and activated_end
                    
                    if activated_start:
                        last_activated_start = activated_start
                        last_activated_count = update_count
        
#                    best_guess = np.argmax(certainty_sums)
                    if activated:
                        center = len(history)-length*2 + (float(activated_start + activated_end) * (length_act//8) + length_act/2) / 2
                        print len(history)
                        print center
                        print center-length//2, center-length//2+length
                        sequences = []
                        for i in range(-8, 9):
                            end = min(center+length//2 + i * 12, len(history))
                            start = end - length
                            sequences.append(history[start:end,channels])
                        test_feed = {inputs: sequences, training: False}
#                        test_output = session1.run(logits, test_feed)[0]
                        test_output = np.mean(np.square(session1.run(logits, test_feed)), axis=0)
                        last_test_output = test_output
                        best_guess = np.argmax(test_output)
                        if time.time() - produced_timestamp > 0.5:
                            produced_timestamp = time.time()
#                            os.system('say ' + str(word_map[best_guess]) + ' &')
                            predictions.append(best_guess)
                            ground_truths.append(None)
                            trigger_predictions.append(None)
                    
#                    if activated and not last_activated or True:
#                        certainty_sums = np.array([0.0] * num_classes)
#                        activated_timestamp = time.time()
#                    if not activated and last_activated:
#                        if time.time() - activated_timestamp >= 0.3 and time.time() - produced_timestamp > 1:
#                        if time.time() - produced_timestamp > 0.5:
#                            produced_timestamp = time.time()
#                            os.system('say ' + str(word_map[best_guess]) + ' &')
#                            predictions.append(best_guess)
    
                    last_activated = activated

                print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i+1) for i in range(8)])
                print str('{:.1f}'.format(count/250.)) + 's\t\t' + '\t'.join(
                    map(lambda (i, x): '{:f}'.format(x) if i in channels else '--\t', enumerate(history[-1])))
                print
                if test_model:
                    if last_test_output is not None:
                        for i in range(len(last_test_output)):
                            n = int(last_test_output[i] * bar_size)
                            print i, '[' + '#' * n + ' ' * (bar_size - n) + ']', last_test_output[i]
#                    print 'Next:', labels[ground_truth_count+1]
                    print
#                    for i in range(len(activation_output)):
#                        for j in range(len(activation_output[i])):
#                            n = int(activation_output[i][j] * bar_size)
#                            print j, '[' + '#' * n + ' ' * (bar_size - n) + ']', activation_output[i][j]
#                        print
                    print ''.join(['#' if x else '-' for x in activation_windows])
                    print ''.join(['#' if activated_start <= i < activated_end else '-' for i in range(len(activation_windows))])
                print 'ACTIVATED' if activated else '---'
        
#                print 'Truth\t\t', ground_truths
##                print 'Trigger\t\t', trigger_predictions
#                print 'Predicted\t', predictions

                truth_string = 'Truth:\t\t'
                trigger_string = 'Trigger:\t'
                predicted_string = 'Predicted:\t'
                for i in range(len(predictions)):
                    x1 = str(ground_truths[i] if ground_truths[i] is not None else '')
                    x2 = str(trigger_predictions[i] if trigger_predictions[i] is not None else '')
                    x3 = str(predictions[i] if predictions[i] is not None else '')
                    max_len = max(map(len, [x1, x2, x3]))
                    truth_string += ' ' + ' ' * (max_len - len(x1)) + x1
                    trigger_string += ' ' + ' ' * (max_len - len(x2)) + x2
                    predicted_string += ' ' + ' ' * (max_len - len(x3)) + x3
                print truth_string
                print trigger_string
                print predicted_string


            displayed = count // step * step

        data.from_file.start('math23_test.txt',
                             on_data, channels=channels, transform_fn=transform_data,
                             history_size=1500, shown_size=1200, # 1200
                             override_step=75, sample_rate=250, speed=10.0, plot=True) # 45
    
#        data.serial.start('/dev/tty.usbserial-DQ007UBV',
#                          on_data, channels=channels, transform_fn=transform_data,
#                          history_size=1500, shown_size=1200, override_step=95) #95