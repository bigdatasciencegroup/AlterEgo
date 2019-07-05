import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
import sys

import data

abs_path = os.path.abspath(os.path.dirname(__file__))

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
    
    return sequence_groups

        
#### Load data

#channels = range(1, 8) # DO NOT CHANGE
#channels = range(4, 8) # DO NOT CHANGE
#channels = range(0, 4) # DO NOT CHANGE
channels = range(0, 3) # DO NOT CHANGE
#channels = range(0, 5) # DO NOT CHANGE
surrounding = 250

labels = [14, 12, 5, 3, 1, 9, 2, 12, 2, 6, 0, 13, 12, 2, 13, 12, 9, 9, 10, 7, 12, 8, 7, 9, 1, 0, 14, 3, 9, 14, 11, 0, 7, 9, 5, 5, 4, 5, 8, 9, 14, 6, 5, 0, 4, 11, 7, 13, 12, 12, 14, 12, 14, 8, 3, 9, 6, 9, 5, 9, 14, 7, 11, 11, 13, 8, 13, 6, 10, 0, 11, 9, 11, 3, 10, 10, 13, 7, 1, 11, 2, 10, 13, 9, 9, 5, 0, 1, 11, 3, 14, 7, 11, 14, 4, 2, 1, 8, 5, 2, 11, 2, 10, 9, 3, 14, 4, 6, 11, 6, 0, 9, 9, 5, 3, 8, 5, 11, 1, 6, 6, 3, 3, 13, 3, 4, 2, 12, 0, 0, 2, 10, 1, 10, 10, 10, 3, 6, 10, 14, 8, 11, 4, 2, 13, 12, 0, 2, 11, 2, 2, 6, 3, 11, 7, 11, 4, 7, 1, 4, 0, 11, 6, 14, 10, 5, 12, 7, 6, 12, 7, 13, 13, 2, 4, 4, 5, 12, 0, 11, 6, 4, 4, 3, 1, 3, 0, 12, 0, 7, 0, 10, 3, 8, 14, 14, 8, 12, 2, 14, 1, 5, 12, 5, 5, 13, 7, 8, 14, 4, 8, 10, 10, 1, 5, 12, 8, 9, 8, 14, 7, 12, 13, 8, 12, 12, 4, 8, 3, 1, 6, 3, 5, 1, 7, 4, 13, 14, 5, 0, 4, 10, 5, 10, 1, 10, 9, 3, 9, 7, 3, 12, 5, 2, 14, 9, 2, 0, 1, 3, 14, 12, 7, 0, 10, 1, 14, 11, 1, 12, 13, 1, 1, 2, 9, 7, 14, 1, 5, 7, 11, 0, 8, 6, 4, 0, 6, 6, 4, 3, 7, 13, 13, 9, 7, 7, 1, 11, 10, 10, 6, 3, 3, 9, 3, 13, 5, 13, 1, 5, 6, 6, 6, 12, 3, 7, 4, 10, 2, 3, 6, 4, 4, 6, 5, 12, 0, 8, 3, 7, 8, 1, 10, 11, 1, 14, 0, 14, 11, 2, 6, 2, 8, 14, 4, 13, 0, 1, 8, 8, 8, 11, 10, 6, 5, 13, 10, 6, 2, 11, 8, 14, 2, 7, 4, 12, 8, 9, 4, 5, 2, 14, 5, 11, 9, 4, 6, 8, 14, 3, 5, 9, 7, 12, 0, 6, 2, 0, 9, 12, 3, 11, 10, 0, 2, 1, 9, 0, 13, 1, 12, 14, 1, 4, 3, 13, 7, 8, 6, 10, 12, 0, 10, 0, 10, 10, 0, 14, 4, 2, 11, 8, 7, 2, 1, 13, 13, 5, 8, 13, 13, 11, 8, 9, 5, 9, 13, 7, 2, 4, 4, 11, 4, 6, 13, 8, 2, 13, 7, 1]

sequence_groups = transform_data([data.flatten(data.process_scrambled(labels, ['math15.txt'], channels=channels, sample_rate=250,
                                                                      surrounding=surrounding, exclude=set([])))])

#silence_group = transform_data(data.process(1, ['math6_silence.txt'], channels=channels, sample_rate=250, surrounding=surrounding))
silence_group = transform_data(data.process_silence_between(['math15.txt'], channels=channels, sample_rate=250, surrounding=surrounding))

sequence_groups = np.array(list(sequence_groups) + list(silence_group))

print len(sequence_groups)
print map(len, sequence_groups)

print
print map(len, sequence_groups[-1])

lengths = map(len, data.get_inputs(sequence_groups)[0])
print min(lengths), np.mean(lengths), max(lengths)

# Split sequence_groups into training and validation data
training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./3)

# Manually selecting different training and validation datasets
#training_sequence_groups = transform_data(data.digits_session_1_dataset())
#validation_sequence_groups = transform_data(data.digits_session_4_dataset())

# Pads or truncates each sequence to length
length = 150 #320, 410 DO NOT CHANGE

training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, 300)
#validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, 300)
#
training_sequence_groups = data.transform.augment_pad_truncate_intervals(training_sequence_groups, length, 4)
#validation_sequence_groups = data.transform.augment_pad_truncate_intervals(validation_sequence_groups, length, 4)

#training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, length)
validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, length)



#test_index = 1
#sequence_groups = data.transform.pad_truncate(sequence_groups, length)
#for i in range(len(sequence_groups[test_index])):
#    plt.plot(sequence_groups[test_index][i][:,3])
#plt.show()
#
#1/0

# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)

# Calculate sample weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
train_weights = class_weights[list(train_labels)]

train_labels = tf.keras.utils.to_categorical(train_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

num_classes = len(training_sequence_groups)

####################
#### Model (MUST BE SAME AS patient_test_serial.py, patient_test_serial_trigger.py, patient_test_serial_silence.py)
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
dropout = tf.layers.dropout(conv4, dropout_rate, training=training)
reshaped = tf.reshape(dropout, [-1, np.prod(dropout.shape[1:])])
fc1 = tf.layers.dense(reshaped, 250, activation=tf.nn.relu)
logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax)

loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
####################


num_epochs = 200
batch_size = 50

num_training_samples = len(train_sequences)
num_validation_samples = len(val_sequences)
num_training_batches = max(1, int(num_training_samples / batch_size))
num_validation_batches = max(1, int(num_validation_samples / batch_size))
start_time = None
last_time = None

# Table display (Ignore this part, it's unnecessarily complicated to make things look pretty)
progress_bar_size = 20
max_batches = max(num_training_batches, num_validation_batches)
layout = [
    dict(name='Ep.', width=len(str(num_epochs)), align='center'),
    dict(name='Batch', width=2*len(str(max_batches))+1, align='center'),
    dict(name='Progress/Timestamp', width=progress_bar_size+2, align='center'),
    dict(name='ETA/Elapsed', width=7, suffix='s', align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Train Loss', width=8, align='center'),
    dict(name='Train Acc', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Val Loss', width=8, align='center'),
    dict(name='Val Acc', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Max Val Acc', width=7, align='center'),
]
since_training = 0
def update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
                 validation_loss=None, validation_accuracy=None, finished=False):
    global last_time
    global since_training
    num_batches = num_training_batches if validation_loss is None else num_validation_batches
    progress = int(math.ceil(progress_bar_size * float(batch) / num_batches))
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
    table.update(epoch + 1, str(batch + 1) + '/' + str(num_batches),
                 progress_string, eta_string, '',
                 training_loss or '--', training_accuracy or '--', '',
                 validation_loss or '--', validation_accuracy or '--', '',
                 max_validation_accuracy if finished else '--')



show_confusion_matrix = True
        
run_accuracies = []
while len(run_accuracies) < 1: #5
    
    good_run = False
    
    saver = tf.train.Saver()
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        table = DynamicConsoleTable(layout)
        table.print_header()

        start_time = time.time()
        last_time = start_time

        # Training/validation loop
        max_validation_accuracy = 0.0
        for epoch in range(num_epochs):
            # Training
            training_loss = 0.0
            training_accuracy = 0.0
            permutation = np.random.permutation(num_training_samples)
            train_sequences = train_sequences[permutation]
            train_labels = train_labels[permutation]
            train_weights = train_weights[permutation]
            train_output = None
            for batch in range(num_training_batches):            
                indices = range(batch * batch_size, (batch + 1) * batch_size)
                if batch == num_training_batches - 1:
                    indices = range(batch * batch_size, num_training_samples)
                batch_sequences = train_sequences[indices]
                batch_labels = train_labels[indices]
                batch_weights = train_weights[indices]

                update_table(epoch, batch, training_loss / (batch_size * max(1, batch)),
                             training_accuracy / (batch_size * max(1, batch)), max_validation_accuracy)

                training_feed = {inputs: batch_sequences, targets: batch_labels,
                                 weights: batch_weights, training: True}
                batch_loss, _, batch_output = session.run([loss, optimizer, logits], training_feed)
                batch_accuracy = session.run(accuracy, training_feed)

                training_loss += batch_loss * len(indices)
                training_accuracy += batch_accuracy * len(indices)
                train_output = batch_output if train_output is None else \
                                        np.concatenate([train_output, batch_output], axis=0)

            training_loss /= num_training_samples
            training_accuracy /= num_training_samples
            
            if training_accuracy > 0.97:
                good_run = True
            if not good_run and epoch + 1 >= 150: #150
                break

            # Validation
            validation_loss = 0.0
            validation_accuracy = 0.0
            val_output = None
            for batch in range(num_validation_batches):         
                indices = range(batch * batch_size, (batch + 1) * batch_size)
                if batch == num_validation_batches - 1:
                    indices = range(batch * batch_size, num_validation_samples)
                batch_sequences = val_sequences[indices]
                batch_labels = val_labels[indices]
                batch_weights = np.ones(len(batch_sequences))

                update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
                             validation_loss / (batch_size * max(1, batch)),
                             validation_accuracy / (batch_size * max(1, batch)))

                validation_feed = {inputs: batch_sequences, targets: batch_labels,
                                   weights: batch_weights, training: False}
                batch_loss, batch_accuracy, batch_output = session.run([loss, accuracy, logits], validation_feed)
                validation_loss += batch_loss * len(indices)
                validation_accuracy += batch_accuracy * len(indices)
                val_output = batch_output if val_output is None else np.concatenate([val_output, batch_output], axis=0)

            validation_loss /= num_validation_samples
            validation_accuracy /= num_validation_samples
            if validation_accuracy > max_validation_accuracy:
                model_name = 'math15_activation_small_model2.ckpt'
                save_path = saver.save(session, os.path.join(abs_path, model_name))
                print ' Model saved:', model_name,
            max_validation_accuracy = max(validation_accuracy, max_validation_accuracy)

            update_table(epoch, batch, training_loss, training_accuracy,
                         max_validation_accuracy, validation_loss, validation_accuracy, finished=True)

            # Also ignore this part
            if show_confusion_matrix:
                predicted = np.argmax(val_output, axis=1)
                actual = np.argmax(val_labels, axis=1)
                val_output = [[] for _ in range(num_classes)]
                val_count_matrix = [[0] * num_classes for _ in range(num_classes)]
                for i in range(len(actual)):
                    val_output[actual[i]].append(predicted[i])
                    val_count_matrix[actual[i]][predicted[i]] += 1
                predicted = np.argmax(train_output, axis=1)
                actual = np.argmax(train_labels, axis=1)
                train_output = [[] for _ in range(num_classes)]
                train_count_matrix = [[0] * num_classes for _ in range(num_classes)]
                for i in range(len(actual)):
                    train_output[actual[i]].append(predicted[i])
                    train_count_matrix[actual[i]][predicted[i]] += 1
                print
                print
                train_max_length = max(2, len(str(max(map(len, training_sequence_groups)))))
                val_max_length = max(2, len(str(max(map(len, validation_sequence_groups)))))
                print 'TRAINING', ' ' * ((num_classes + 1) * (train_max_length + 1) + 1), '\t\t\t', 'VALIDATION'
                print ' ' * (train_max_length - 2) + 'Predicted ', ''.join(map(
                        lambda x: str(x) + ' ' * (train_max_length - len(str(x)) + 1), range(num_classes))), '\t\t\t', \
                        ' ' * (val_max_length - 2) + 'Predicted ', ''.join(map(
                        lambda x: str(x) + ' ' * (val_max_length - len(str(x)) + 1), range(num_classes)))
                print 'Actual\t', '-' * ((num_classes + 1) * (train_max_length + 1) + 1), '\t\t\t', \
                        'Actual\t', '-' * ((num_classes + 1) * (val_max_length + 1) + 1)
                for i in range(num_classes):
                    print ' '*(5-len(str(i))), i, '\t|' + ' ' * (train_max_length - 1), \
                        ''.join(map(lambda x: (str(x) if x else '-') + ' ' * (train_max_length - len(str(x)) + 1), \
                                    train_count_matrix[i])) + '| ', \
                        str.format('{0:.5f}', np.mean(np.array(train_output[i]) == i)), \
                        '\t\t', ' '*(5-len(str(i))), i, '\t|' + ' ' * (val_max_length - 1), \
                        ''.join(map(lambda x: (str(x) if x else '-') + ' ' * (val_max_length - len(str(x)) + 1), \
                                    val_count_matrix[i])) + '| ', \
                        str.format('{0:.5f}', np.mean(np.array(val_output[i]) == i))
                print '\t', '-' * ((num_classes + 1) * (train_max_length + 1) + 1), '\t\t\t', \
                        '\t', '-' * ((num_classes + 1) * (val_max_length + 1) + 1)

            reprint_header = ((epoch+1) % 10 == 0 or show_confusion_matrix) and epoch < num_epochs - 1
            table.finalize(divider=not reprint_header)
            if reprint_header:
                table.print_header()
        
        if good_run:
            run_accuracies.append(max_validation_accuracy)
            
            model_name = 'math15_activation_small_model2.ckpt'
            save_path = saver.save(session, os.path.join(abs_path, model_name))
            print ' Model saved:', model_name,
    
    print
    print 'RUN ACCURACIES:'
    print run_accuracies
    print np.mean(run_accuracies)
    print
