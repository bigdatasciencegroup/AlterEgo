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
#    
    return sequence_groups

        
#### Load data
# sequence_groups[i] contains data for class i
# 4-dimensional data structure: (class, sequence_num, timestep, channel_num)

sequence_groups = transform_data(data.digits_session_4_dataset(channels=range(1, 8)))
training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./6)

# Pads or truncates each sequence to length
length = 300 # 2000, 300,600
training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, length)
validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, length)

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


num_channels = len(train_sequences[0][0])

window_size = 300 # 2000

dropout_rate = 0.4

inputs = tf.placeholder(tf.float32,[None, window_size, num_channels])
targets = tf.placeholder(tf.float32,[None, window_size, num_channels])
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
fc1 = tf.layers.dense(reshaped, 125, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, 250, activation=tf.nn.relu)
#fc2 = tf.layers.dense(reshaped, 250, activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2, 500, activation=tf.nn.relu)
outputs = tf.layers.dense(fc3, window_size * num_channels)
outputs = tf.reshape(outputs, [-1, window_size, num_channels])



batch_size = 50 #50

num_training_samples = len(train_sequences)
num_validation_samples = len(val_sequences)
num_training_batches = max(1, int(num_training_samples / batch_size))
num_validation_batches = max(1, int(num_validation_samples / batch_size))

#print np.shape(train_sequences)
#print np.shape(val_sequences)

saver = tf.train.Saver()
with tf.Session() as session:
    tf.global_variables_initializer().run()
    saver.restore(session, 'autoencoder_digits_300_model.ckpt')

    training_output = None
    for batch in range(num_training_batches):         
        indices = range(batch * batch_size, (batch + 1) * batch_size)
        if batch == num_training_batches - 1:
            indices = range(batch * batch_size, num_training_samples)
        batch_sequences = train_sequences[indices]

        feed = {inputs: batch_sequences, training: False}
        batch_output = session.run(fc2, feed)
        training_output = batch_output if training_output is None else \
                            np.concatenate([training_output, batch_output], axis=0)
            
    validation_output = None
    for batch in range(num_validation_batches):         
        indices = range(batch * batch_size, (batch + 1) * batch_size)
        if batch == num_validation_batches - 1:
            indices = range(batch * batch_size, num_validation_samples)
        batch_sequences = val_sequences[indices]

        feed = {inputs: batch_sequences, training: False}
        batch_output = session.run(fc2, feed)
        validation_output = batch_output if validation_output is None else \
                            np.concatenate([validation_output, batch_output], axis=0)
        
    train_sequences = training_output
    val_sequences = validation_output
            
#print np.shape(train_sequences)
#print np.shape(val_sequences)


tf.reset_default_graph()


#### Model
learning_rate = 1e-4
dropout_rate = 0.4

#inputs = tf.placeholder(tf.float32,[None, length, np.shape(train_sequences[0])[1]]) #[batch_size,timestep,features]
inputs = tf.placeholder(tf.float32,[None, 250])
targets = tf.placeholder(tf.int32, [None, num_classes])
weights = tf.placeholder(tf.float32, [None])

fc1 = tf.layers.dense(inputs, 1024, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2, 1024, activation=tf.nn.relu)
logits = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax)

loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



num_epochs = 20000
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
    num_batches = num_training_batches if validation_loss is None or finished else num_validation_batches
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
    table.update(epoch + 1, str(batch + 1 if not finished else num_batches) + '/' + str(num_batches),
                 progress_string, eta_string, '',
                 training_loss or '--', training_accuracy or '--', '',
                 validation_loss or '--', validation_accuracy or '--', '',
                 max_validation_accuracy if finished else '--')



show_confusion_matrix = True

#good_run = False
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
        for batch in range(num_training_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_training_batches - 1:
                indices = range(batch * batch_size, num_training_samples)
            batch_sequences = train_sequences[indices]
            batch_labels = train_labels[indices]
            batch_weights = train_weights[indices]
            
            update_table(epoch, batch, training_loss / (batch_size * max(1, batch)),
                         training_accuracy / (batch_size * max(1, batch)), max_validation_accuracy)
                        
            training_feed = {inputs: batch_sequences, targets: batch_labels, weights: batch_weights}
            batch_loss, _ = session.run([loss, optimizer], training_feed)
            batch_accuracy = session.run(accuracy, training_feed)
                        
            training_loss += batch_loss * len(indices)
            training_accuracy += batch_accuracy * len(indices)
            
        training_loss /= num_training_samples
        training_accuracy /= num_training_samples
        
        # Validation
        validation_loss = 0.0
        validation_accuracy = 0.0
        output = None
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
            
            validation_feed = {inputs: batch_sequences, targets: batch_labels, weights: batch_weights}
            batch_loss, batch_accuracy, batch_output = session.run([loss, accuracy, logits], validation_feed)
            validation_loss += batch_loss * len(indices)
            validation_accuracy += batch_accuracy * len(indices)
            output = batch_output if output is None else np.concatenate([output, batch_output], axis=0)
            
        validation_loss /= num_validation_samples
        validation_accuracy /= num_validation_samples
        max_validation_accuracy = max(validation_accuracy, max_validation_accuracy)
        
#        if training_accuracy >= 0.90:
#            good_run = True
#        if epoch +1 >= 100 and not good_run:
#            print '\n\n'
#            sys.exit()
                
        update_table(epoch, batch, training_loss, training_accuracy,
                     max_validation_accuracy, validation_loss, validation_accuracy, finished=True)
        
        if show_confusion_matrix:
            predicted = np.argmax(output, axis=1)
            actual = np.argmax(val_labels, axis=1)
            output = [[] for _ in range(num_classes)]
            count_matrix = [[0] * num_classes for _ in range(num_classes)]
            for i in range(len(actual)):
                output[actual[i]].append(predicted[i])
                count_matrix[actual[i]][predicted[i]] += 1
            print
            print
            max_length = max(2, len(str(max(map(len, validation_sequence_groups)))))
            print ' ' * (max_length - 2) + 'Predicted ', ''.join(map(
                    lambda x: str(x) + ' ' * (max_length - len(str(x)) + 1), range(num_classes)))
            print 'Actual\t', '-' * ((num_classes + 1) * (max_length + 1) + 1)
            for i in range(num_classes):
                print ' '*(5-len(str(i))), i, '\t|' + ' ' * (max_length - 1), \
                    ''.join(map(lambda x: (str(x) if x else '-') + ' ' * (max_length - len(str(x)) + 1), \
                                count_matrix[i])) + '| ', np.mean(np.array(output[i]) == i)
        
        reprint_header = ((epoch+1) % 10 == 0 or show_confusion_matrix) and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()
