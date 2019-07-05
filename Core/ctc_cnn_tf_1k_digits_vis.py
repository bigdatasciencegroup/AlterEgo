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

import data

#length = 2800 #600 #2800
    
# Load data and apply transformations
sequence_groups = data.digits_sequences_session_2_dataset(include_surrounding=False, channels=range(4, 8))

#sequence_groups = data.transform.default_transform(sequence_groups)

sample_rate = 1000 #250 #1000

drift_low_frequency = 0.5 #0.5
sequence_groups = data.transform.subtract_initial(sequence_groups)
sequence_groups = data.transform.highpass_filter(sequence_groups, drift_low_frequency, sample_rate)
sequence_groups = data.transform.subtract_mean(sequence_groups)

notch1 = 60 #59.9
num_times = 3 #2
freqs = map(int, map(round, np.arange(1, sample_rate/(2. * notch1)) * notch1))
for _ in range(num_times):
    for f in reversed(freqs):
        sequence_groups = data.transform.notch_filter(sequence_groups, f, sample_rate)

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

ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
ricker_subtraction_multiplier = 2
sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

period = 250 * sample_rate // 250
sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)

#low = 0.5 #0.5
#high = 8 #8
#order = 1
#bandpass_filtered = data.transform.bandpass_filter(sequence_groups, low, high, sample_rate, order=order)


# Split into training and validation data
training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./6)
#training_sequence_groups = list(sequence_groups[:4])
#training_sequence_groups.append([])
#training_sequence_groups.append([])
#training_sequence_groups = np.array(training_sequence_groups)
#validation_sequence_groups = list(sequence_groups[4:])
#validation_sequence_groups.insert(0, [])
#validation_sequence_groups.insert(0, [])
#validation_sequence_groups.insert(0, [])
#validation_sequence_groups.insert(0, [])
#validation_sequence_groups = np.array(validation_sequence_groups)

#training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, length)
#validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, length)

# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)

# Calculate sample weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
train_weights = class_weights[list(train_labels)]

#train_labels = tf.keras.utils.to_categorical(train_labels)
#val_labels = tf.keras.utils.to_categorical(val_labels)
label_map = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]]
#label_map = [[0, 1, 2, 3], [1, 2, 3, 0], [3, 0, 1, 2], [0, 3, 1, 2], [3, 1, 2, 0], [1, 2, 0, 3]]
train_labels = np.array(map(lambda i: label_map[i], train_labels))
val_labels = np.array(map(lambda i: label_map[i], val_labels))

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

num_classes = len(np.unique(reduce(lambda a,b: a+b, label_map))) + 1


learning_rate = 5e-4 #5e-4 #1e-3
dropout_rate = 0.4

cnn_window_size = 1250 * sample_rate // 1000 #1250
cnn_window_stride = 250 * sample_rate // 1000 #250
num_channels = 4

inputs = tf.placeholder(tf.float32,[None, None, cnn_window_size, num_channels]) #[batch_size,timestep,window,features]
targets = tf.sparse_placeholder(tf.int32)
sequence_lengths = tf.placeholder(tf.int32, [None])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)
batch_size = tf.shape(inputs)[0]
max_timesteps = tf.shape(inputs)[1]

vis_inputs = tf.placeholder(tf.float32, [None, cnn_window_size, num_channels])
vis_targets = tf.placeholder(tf.int32, [None, num_classes])

reshaped = tf.reshape(inputs, [-1, cnn_window_size, num_channels])

conv1 = tf.layers.conv1d(reshaped, 512, 50, activation=tf.nn.relu, padding='valid', name='conv1')
conv1 = tf.layers.max_pooling1d(conv1, 2, strides=2)
conv2 = tf.layers.conv1d(conv1, 512, 25, activation=tf.nn.relu, padding='valid', name='conv2')
conv2 = tf.layers.max_pooling1d(conv2, 2, strides=2)
conv3 = tf.layers.conv1d(conv2, 512, 12, activation=tf.nn.relu, padding='valid', name='conv3')
conv3 = tf.layers.max_pooling1d(conv3, 2, strides=2)
conv4 = tf.layers.conv1d(conv3, 512, 12, activation=tf.nn.relu, padding='valid', name='conv4')
conv4 = tf.layers.max_pooling1d(conv4, 2, strides=2)
dropout = tf.layers.dropout(conv4, dropout_rate, training=training)

vis_conv1 = tf.layers.conv1d(vis_inputs, 512, 50, activation=tf.nn.relu, padding='valid', name='conv1', reuse=True)
vis_conv1 = tf.layers.max_pooling1d(vis_conv1, 2, strides=2)
vis_conv2 = tf.layers.conv1d(vis_conv1, 512, 25, activation=tf.nn.relu, padding='valid', name='conv2', reuse=True)
vis_conv2 = tf.layers.max_pooling1d(vis_conv2, 2, strides=2)
vis_conv3 = tf.layers.conv1d(vis_conv2, 512, 12, activation=tf.nn.relu, padding='valid', name='conv3', reuse=True)
vis_conv3 = tf.layers.max_pooling1d(vis_conv3, 2, strides=2)
vis_conv4 = tf.layers.conv1d(vis_conv3, 512, 12, activation=tf.nn.relu, padding='valid', name='conv4', reuse=True)
vis_conv4 = tf.layers.max_pooling1d(vis_conv4, 2, strides=2)
vis_dropout = tf.layers.dropout(vis_conv4, dropout_rate, training=False)



# EXPLORATORY (global average pooling)
#dropout = tf.reduce_mean(dropout, axis=1)

reshaped = tf.reshape(dropout, [-1, np.prod(dropout.shape[1:])])
vis_reshaped = tf.reshape(vis_dropout, [-1, np.prod(vis_dropout.shape[1:])])
#fc_size = 1024
#fc1 = tf.layers.dense(reshaped, fc_size, activation=tf.nn.relu)
#reshaped = tf.reshape(fc1, [batch_size, max_timesteps, fc_size])

#lstm_hidden_size = 256
#lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
#dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, 1.0-dropout_rate)
#lstm1b = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
#dropout1b = tf.nn.rnn_cell.DropoutWrapper(lstm1b, 1.0-dropout_rate)
#lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
#dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, 1.0-dropout_rate)
#lstm2b = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
#dropout2b = tf.nn.rnn_cell.DropoutWrapper(lstm2b, 1.0-dropout_rate)
#
#forward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout2])
#backward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout2b])
#
#outputs, states = tf.nn.bidirectional_dynamic_rnn(forward_stack, backward_stack, inputs, dtype=tf.float32)
#outputs = tf.concat(outputs, 2)
#
#reshaped = tf.reshape(outputs, [-1, 2 * lstm_hidden_size])

#reshaped = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu,
#            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(2 * lstm_hidden_size * 1024))))

fc_size = 512 #1024 #512

reshaped = tf.layers.dense(reshaped, fc_size, activation=tf.nn.relu,
    kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(int(np.prod(dropout.shape[1:])) * fc_size))), name='fc1')
reshaped = tf.layers.dense(reshaped, fc_size, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(fc_size * fc_size))),
                           name='fc2')
logits = tf.layers.dense(reshaped, num_classes,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(fc_size * num_classes))),
                         name='fc3')

vis_reshaped = tf.layers.dense(vis_reshaped, fc_size, activation=tf.nn.relu, name='fc1', reuse=True)
vis_reshaped = tf.layers.dense(vis_reshaped, fc_size, activation=tf.nn.relu, name='fc2', reuse=True)
vis_logits = tf.layers.dense(vis_reshaped, num_classes, name='fc3', reuse=True)


logits = tf.reshape(logits, [batch_size, -1, num_classes])

vis_logits = tf.reshape(vis_logits, [tf.shape(vis_inputs)[0], -1, num_classes])
vis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vis_logits, labels=vis_targets))
vis_gradients = tf.gradients(vis_loss, vis_inputs)
vis_gradients /= tf.sqrt(tf.reduce_mean(tf.square(vis_gradients))) + 1e-5

logits = tf.transpose(logits, [1, 0, 2])

#loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))
loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=sequence_lengths, time_major=True)
loss = tf.reduce_mean(tf.multiply(loss, weights))

# L2 regularization
#l2 = 0.005 * sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()\
#                          if 'noreg' not in tf_var.name.lower() and 'bias' not in tf_var.name.lower()])
#loss += l2

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

#opt = tf.train.RMSPropOptimizer(learning_rate)
#gradients = opt.compute_gradients(loss)
#clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
#optimizer = opt.apply_gradients(clipped_gradients)

#decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sequence_lengths)
ctc_output = tf.nn.ctc_beam_search_decoder(logits, sequence_lengths)
decoded, log_prob = ctc_output[0][0], ctc_output[1][0]

#error = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets, normalize=True))
error = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), targets, normalize=True))


num_epochs = 20000 #200
batch_size = 50 #50

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
    if finished and (epoch + 1) % 10 == 0:
        print
        print 'training_losses =', training_losses
        print
        print 'training_errors =', training_errors
        print
        print 'validation_losses =', validation_losses
        print
        print 'validation_errors =', validation_errors
            
def sparsify(labels):
    indices = []
    values = []
    for n, seq in enumerate(labels):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(labels), max(map(len, labels))], dtype=np.int64)
    return indices, values, shape

def window_split(sequence, window, stride=None, step=1):
    if stride is None: stride = window
    return np.array([sequence[i:i+window:step] for i in range(0, len(sequence)-window+1, stride)])

with tf.Session() as session:
    tf.global_variables_initializer().run()
    
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    start_time = time.time()
    last_time = start_time
    
    min_validation_error = float('inf')
    for epoch in range(num_epochs):
        training_loss = 0.0
        training_error = 0.0
        permutation = np.random.permutation(num_training_samples)
        train_sequences = train_sequences[permutation]
        train_labels = train_labels[permutation]
        train_weights = train_weights[permutation]
        training_decoded = []
        for batch in range(num_training_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_training_batches - 1:
                indices = range(batch * batch_size, num_training_samples)
            batch_sequences = train_sequences[indices]
#            print np.shape(batch_sequences)
            tmp_lengths = map(len, batch_sequences)
            tmp = np.array(map(lambda seq: window_split(seq, cnn_window_size, stride=cnn_window_stride),
                               batch_sequences))
            batch_lengths = map(len, tmp)
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(tmp_lengths), position=0.0)
            batch_sequences = np.array(map(lambda seq: window_split(seq, cnn_window_size, stride=cnn_window_stride),
                                           batch_sequences))
#            print np.shape(batch_sequences)
            batch_labels = train_labels[indices]
            batch_weights = train_weights[indices]
            sparse_batch_labels = sparsify(batch_labels)
            
            update_table(epoch, batch, training_loss / (batch_size * max(1, batch)),
                         training_error / (batch_size * max(1, batch)), min_validation_error)
                        
            training_feed = {inputs: batch_sequences, targets: sparse_batch_labels,
                             sequence_lengths: batch_lengths, weights: batch_weights, training: True}
            batch_loss, _ = session.run([loss, optimizer], training_feed)
            batch_decoded, batch_error = session.run([decoded, error], training_feed)
                        
            training_loss += batch_loss * len(indices)
            training_error += batch_error * len(indices)
#            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded[0], default_value=-1).eval()
            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded, default_value=-1).eval()
            for seq in batch_decoded:
                training_decoded.append(seq)
            
        training_loss /= num_training_samples
        training_error /= num_training_samples
                
        validation_loss = 0.0
        validation_error = 0.0
        validation_decoded = []
        for batch in range(num_validation_batches):         
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_validation_batches - 1:
                indices = range(batch * batch_size, num_validation_samples)
            batch_sequences = val_sequences[indices]
#            print np.shape(batch_sequences)
            tmp_lengths = map(len, batch_sequences)
            tmp = np.array(map(lambda seq: window_split(seq, cnn_window_size, stride=cnn_window_stride),
                               batch_sequences))
            batch_lengths = map(len, tmp)
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(tmp_lengths), position=0.0)
            batch_sequences = np.array(map(lambda seq: window_split(seq, cnn_window_size, stride=cnn_window_stride),
                                           batch_sequences))
#            print np.shape(batch_sequences)
            batch_labels = val_labels[indices]
            batch_weights = np.ones(len(indices))
            sparse_batch_labels = sparsify(batch_labels)
                        
            update_table(epoch, batch, training_loss, training_error, min_validation_error,
                         validation_loss / (batch_size * max(1, batch)),
                         validation_error / (batch_size * max(1, batch)))
            
            validation_feed = {inputs: batch_sequences, targets: sparse_batch_labels,
                               sequence_lengths: batch_lengths, weights: batch_weights, training: False}
            batch_output = session.run([loss, error, decoded, log_prob], validation_feed)
            batch_loss, batch_error, batch_decoded, batch_log_prob = batch_output
            validation_loss += batch_loss * len(indices)
            validation_error += batch_error * len(indices)
#            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded[0], default_value=-1).eval()
            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded, default_value=-1).eval()
            for seq in batch_decoded:
                validation_decoded.append(seq)
        validation_loss /= num_validation_samples
        validation_error /= num_validation_samples
        min_validation_error = min(validation_error, min_validation_error)
        
        avg_kernel = normalize_kernel([1.0] * (sample_rate // 20))
        def deprocess(x):
#            x -= np.mean(x)
#            x /= (np.std(x) + 1e-5)
            x = data.transform.correlate(x, avg_kernel)
#            x *= 0.1
#            x += 0.5
#            x = np.clip(x, 0, 1)
#            x *= 255
##            x = np.transpose(x, (1, 2, 0))
#            x = np.clip(x, 0, 255).astype('uint8')
            return x
        test_index = 1
        num_copies = 10
        for test_index in range(num_classes-1):
            if (epoch+1) % 1 == 0 and (epoch+1) > 1:
                test_input = np.random.rand((num_classes-1)*num_copies, cnn_window_size, num_channels) * 2 - 1.0
                for i in range(150):
                    test_targets = np.zeros(((num_classes-1)*num_copies, num_classes))
                    for c in range(num_classes-1):
                        test_targets[c*num_copies:(c+1)*num_copies,c] = 1.0
                    gradients_output = session.run(vis_gradients, {vis_inputs: test_input, vis_targets: test_targets})
                    test_input += gradients_output[0] * 0.02
    #                test_input += (np.random.rand(num_classes-1, cnn_window_size, num_channels) - 0.5) * 0.02
                    if (i+1) % 15 == 0 or i == 0:
                        plt.gcf().clear()
                        plt.plot(deprocess(np.mean(test_input[test_index*num_copies:(test_index+1)*num_copies,:,:], axis=0)))
                        plt.pause(0.00001)
                        print i+1
                    if i == 149:
                        np.save('visualized_' + str(test_index),
                                np.mean(test_input[test_index*num_copies:(test_index+1)*num_copies,:,:], axis=0))
        
        update_table(epoch, batch, training_loss, training_error,
                     min_validation_error, validation_loss, validation_error, finished=True)
        print
        print 'Training:'
        for i in range(min(5, len(training_decoded))):
            print train_labels[i], ' => ', training_decoded[i][np.where(training_decoded[i] > -1)]
        print 'Validation:'
        for i in range(min(5, len(validation_decoded))):
            print val_labels[i], ' => ', validation_decoded[i][np.where(validation_decoded[i] > -1)]
        
        reprint_header = (epoch+1) % 10 == 0 and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()
