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

import data

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
    
# Load data and apply transformations
sequence_groups = transform_data(data.words_20_25_sentences_dataset(include_surrounding=False, channels=range(1,8)))
#sequence_groups = sequence_groups[:,:2]

lens = map(len, data.get_inputs(sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)


# Split into training and validation data
#training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./5)

def train_test_split(sequence_groups, test_indices):
    training_sequence_groups = []
    validation_sequence_groups = []
    test_indices = set(test_indices)
    for i in range(len(sequence_groups)):
        if i in test_indices:
            validation_sequence_groups.append(sequence_groups[i])
            training_sequence_groups.append([])
        else:
            training_sequence_groups.append(sequence_groups[i])
            validation_sequence_groups.append([])
    return training_sequence_groups, validation_sequence_groups
            
training_sequence_groups, validation_sequence_groups = train_test_split(sequence_groups, range(2, 25, 5))
print map(len, training_sequence_groups)
print map(len, validation_sequence_groups)

#training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, length)
#validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, length)

# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)

# Calculate sample weights
#class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
#train_weights = class_weights[list(train_labels)]
train_weights = np.ones(len(train_labels))

#train_labels = tf.keras.utils.to_categorical(train_labels)
#val_labels = tf.keras.utils.to_categorical(val_labels)

'''
of the feeling that i
[17, 11, 19, 12, 0]
because i am tired of
[18, 0, 1, 5, 17]
that what you want the
[12, 16, 14, 6, 11]
hot because you need what
[3, 18, 14, 7, 16]
cold water where am feeling
[2, 9, 15, 1, 19]
where thank want cold food
[15, 13, 6, 2, 8]
hungry that need of hello
[4, 12, 7, 17, 10]
am where i want you
[1, 15, 0, 6, 14]
cold because the hot food
[2, 18, 11, 3, 8]
what water hello thank hungry
[16, 9, 10, 13, 4]
thank you where that tired
[13, 14, 15, 12, 5]
because of what i need
[18, 17, 16, 0, 7]
am the cold feeling hungry
[1, 11, 2, 19, 4]
water want tired hello food
[9, 6, 5, 10, 8]
of where hot that the
[17, 15, 3, 12, 11]
need you what am i
[7, 14, 16, 1, 0]
want water food need thank
[6, 9, 8, 7, 13]
water hungry because cold hello
[9, 4, 18, 2, 10]
feeling tired that you hot
[19, 5, 12, 14, 3]
am because food of water
[1, 18, 8, 17, 9]
cold i what the food
[2, 0, 16, 11, 8]
where want hungry tired thank
[15, 6, 4, 5, 13]
you hello hot water feeling
[14, 10, 3, 9, 19]
of feeling the need that
[17, 19, 11, 7, 12]
because what need i thank
[18, 16, 7, 0, 13]

Official label order:
['am', 'because', 'cold', 'feeling', 'food', 'hello', 'hot', 'hungry', 'i', 'need', 'of', 'thank', 'that', 'the', 'tired', 'want', 'water', 'what', 'where', 'you']
'''
words = np.array(['am', 'because', 'cold', 'feeling', 'food', 'hello', 'hot', 'hungry', 'i', 'need', 'of', 'thank', 'that', 'the', 'tired', 'want', 'water', 'what', 'where', 'you'])
label_map = [
    [17, 11, 19, 12, 0],
    [18, 0, 1, 5, 17],
    [12, 16, 14, 6, 11],
    [3, 18, 14, 7, 16],
    [2, 9, 15, 1, 19],
    [15, 13, 6, 2, 8],
    [4, 12, 7, 17, 10],
    [1, 15, 0, 6, 14],
    [2, 18, 11, 3, 8],
    [16, 9, 10, 13, 4],
    [13, 14, 15, 12, 5],
    [18, 17, 16, 0, 7],
    [1, 11, 2, 19, 4],
    [9, 6, 5, 10, 8],
    [17, 15, 3, 12, 11],
    [7, 14, 16, 1, 0],
    [6, 9, 8, 7, 13],
    [9, 4, 18, 2, 10],
    [19, 5, 12, 14, 3],
    [1, 18, 8, 17, 9],
    [2, 0, 16, 11, 8],
    [15, 6, 4, 5, 13],
    [14, 10, 3, 9, 19],
    [17, 19, 11, 7, 12],
    [18, 16, 7, 0, 13],
]
train_labels = np.array(map(lambda i: label_map[i], train_labels))
val_labels = np.array(map(lambda i: label_map[i], val_labels))

bigram_count_map = {}
for i in range(len(label_map)):
    for j in range(len(label_map[i])-1):
        key = tuple(label_map[i][j:j+2])
        bigram_count_map[key] = bigram_count_map.get(key, 0) + 1
trigram_count_map = {}
for i in range(len(label_map)):
    for j in range(len(label_map[i])-2):
        key = tuple(label_map[i][j:j+3])
        trigram_count_map[key] = trigram_count_map.get(key, 0) + 1
fourgram_count_map = {}
for i in range(len(label_map)):
    for j in range(len(label_map[i])-3):
        key = tuple(label_map[i][j:j+4])
        fourgram_count_map[key] = fourgram_count_map.get(key, 0) + 1
fivegram_count_map = {}
for i in range(len(label_map)):
    for j in range(len(label_map[i])-4):
        key = tuple(label_map[i][j:j+5])
        fivegram_count_map[key] = fivegram_count_map.get(key, 0) + 1
unigram_count_map = {unigram: count for (unigram, count) in\
                     zip(*np.unique(np.concatenate(label_map), return_counts=True))}
print unigram_count_map
print bigram_count_map
print len(bigram_count_map)
print trigram_count_map
print len(trigram_count_map)

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

num_classes = len(np.unique(reduce(lambda a,b: a+b, label_map))) + 1


learning_rate = 1e-4 #5e-4 #1e-3
dropout_rate = 0.4

sample_rate = 250
cnn_window_size = 1000 * sample_rate // 1000 #1000 #600 #1250
cnn_window_stride = 250 * sample_rate // 1000 #250 #100 #250
num_channels = 7

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

conv1 = tf.layers.conv1d(reshaped, 512, 12, activation=tf.nn.relu, padding='valid', name='conv1') # 3
conv1 = tf.layers.max_pooling1d(conv1, 2, strides=2)
conv2 = tf.layers.conv1d(conv1, 512, 6, activation=tf.nn.relu, padding='valid', name='conv2')
conv2 = tf.layers.max_pooling1d(conv2, 2, strides=2)
conv3 = tf.layers.conv1d(conv2, 512, 3, activation=tf.nn.relu, padding='valid', name='conv3')
conv3 = tf.layers.max_pooling1d(conv3, 2, strides=2)
conv4 = tf.layers.conv1d(conv3, 512, 3, activation=tf.nn.relu, padding='valid', name='conv4')
conv4 = tf.layers.max_pooling1d(conv4, 2, strides=2)
dropout = tf.layers.dropout(conv4, dropout_rate, training=training)

vis_conv1 = tf.layers.conv1d(vis_inputs, 512, 12, activation=tf.nn.relu, padding='valid', name='conv1', reuse=True)
vis_conv1 = tf.layers.max_pooling1d(vis_conv1, 2, strides=2)
vis_conv2 = tf.layers.conv1d(vis_conv1, 512, 6, activation=tf.nn.relu, padding='valid', name='conv2', reuse=True)
vis_conv2 = tf.layers.max_pooling1d(vis_conv2, 2, strides=2)
vis_conv3 = tf.layers.conv1d(vis_conv2, 512, 3, activation=tf.nn.relu, padding='valid', name='conv3', reuse=True)
vis_conv3 = tf.layers.max_pooling1d(vis_conv3, 2, strides=2)
vis_conv4 = tf.layers.conv1d(vis_conv3, 512, 3, activation=tf.nn.relu, padding='valid', name='conv4', reuse=True)
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

fc_size = 1024 #512

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
ctc_output = tf.nn.ctc_beam_search_decoder(logits, sequence_lengths, top_paths=10)
decoded, log_prob = ctc_output[0], ctc_output[1]

error = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets, normalize=True))

decoded_selected = tf.sparse_placeholder(tf.int32)
#decoded_selected = tf.placeholder(tf.int32, [None, None])
#sparse_decoded_selected = tf.contrib.layers.dense_to_sparse(decoded_selected, eos_token=-1)
error_selected = tf.reduce_mean(tf.edit_distance(decoded_selected, targets, normalize=True))


num_epochs = 20000 #200
batch_size = 20 #20 #50

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
#    if finished and (epoch + 1) % 10 == 0:
#        print
#        print 'training_losses =', training_losses
#        print
#        print 'training_errors =', training_errors
#        print
#        print 'validation_losses =', validation_losses
#        print
#        print 'validation_errors =', validation_errors
            
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
    errors = []
    error_selecteds = []
    for epoch in range(num_epochs):
        training_loss = 0.0
        training_error = 0.0
        permutation = np.random.permutation(num_training_samples)
        train_sequences = train_sequences[permutation]
        train_labels = train_labels[permutation]
        train_weights = train_weights[permutation]
        training_decoded = []
        training_log_probs = []
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
            batch_loss, _, batch_decoded, batch_error, batch_log_prob = \
                                    session.run([loss, optimizer, decoded, error, log_prob], training_feed)
            
            training_loss += batch_loss * len(indices)
            training_error += batch_error * len(indices)
#            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded[0], default_value=-1).eval()
            batch_decoded = map(lambda x: tf.sparse_tensor_to_dense(x, default_value=-1).eval(), batch_decoded)
            for i in range(len(batch_decoded[0])):
                training_decoded.append([])
                for j in range(len(batch_decoded)):
                    training_decoded[-1].append(batch_decoded[j][i])
            for log_probs in batch_log_prob:
                training_log_probs.append(log_probs)
            
        training_loss /= num_training_samples
        training_error /= num_training_samples
                
        validation_loss = 0.0
        validation_error = 0.0
        validation_error_selected = 0.0
        permutation = np.random.permutation(num_validation_samples)
        val_sequences = val_sequences[permutation]
        val_labels = val_labels[permutation]
        validation_decoded = []
        validation_log_probs = []
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
            batch_loss, batch_error, batch_decoded, batch_log_prob = \
                                            session.run([loss, error, decoded, log_prob], validation_feed)
            validation_loss += batch_loss * len(indices)
            validation_error += batch_error * len(indices)
            
            batch_decoded = map(lambda x: tf.sparse_tensor_to_dense(x, default_value=-1).eval(), batch_decoded)
            
            # Calculate the most likely path based on language model
#            for i in range(len(batch_decoded)):
#                tmp = filter(lambda x: x > -1, batch_decoded[i][0])
#                print tmp
#                print batch_log_prob[0][i]
#                for j in range(len(tmp)-1):
#                    key = tuple(tmp[j:j+2])
#                    print key, 1.0 if key in bigram_count_map else 0.0
#                for j in range(len(tmp)-2):
#                    key = tuple(tmp[j:j+3])
#                    print key, 1.0 if key in trigram_count_map else 0.0
                    
            # modifies probabilities
            for i in range(len(batch_decoded)):
                for j in range(len(batch_decoded[i])):
                    tmp = filter(lambda x: x > -1, batch_decoded[i][j])
#                    print tmp
#                    print batch_log_prob[j][i]
                    for k in range(len(tmp)-1):
                        key = tuple(tmp[k:k+2])
                        batch_log_prob[j][i] += np.log(1.0 if key in bigram_count_map else 0.5)
                    for k in range(len(tmp)-2):
                        key = tuple(tmp[k:k+3])
                        batch_log_prob[j][i] += np.log(1.0 if key in trigram_count_map else 0.5)
                    for k in range(len(tmp)-3):
                        key = tuple(tmp[k:k+4])
                        batch_log_prob[j][i] += np.log(1.0 if key in fourgram_count_map else 0.5)
                    for k in range(len(tmp)-4):
                        key = tuple(tmp[k:k+5])
                        batch_log_prob[j][i] += np.log(1.0 if key in fivegram_count_map else 0.5)
                
            batch_decoded_selected = batch_decoded[0]
            batch_decoded_selected = map(list, map(lambda x: x[np.where(x > -1)], batch_decoded_selected))
            
#            print
#            print batch_decoded_selected
            
            batch_decoded_selected = []
            for i in range(len(batch_decoded[0])):
                pairs = [(batch_log_prob[i][j], batch_decoded[j][i]) for j in range(len(batch_decoded))]
                pairs = sorted(pairs)
#                while len(pairs) > 1 and not len(pairs[-1][1]):
#                    pairs.pop()
                pairs = pairs[::-1]
                selected = pairs[0][1][np.where(pairs[0][1] > -1)]
                batch_decoded_selected.append(list(selected))
#            max_length = max(map(len, batch_decoded_selected))
#            for i in range(len(batch_decoded_selected)):
#                while len(batch_decoded_selected[i]) < max_length:
#                    batch_decoded_selected[i].append(-1)

#            print batch_decoded_selected
            
            sparse_indices, value, shape = sparsify(batch_decoded_selected)
            if not len(sparse_indices):
                sparse_indices = np.zeros((0, 2))
            
            selected_feed = {decoded_selected: (sparse_indices, value, shape), targets: sparse_batch_labels}
            batch_error_selected = session.run(error_selected, selected_feed)
            validation_error_selected += batch_error_selected * len(indices)
            
#            print batch_error, batch_error_selected
            
            for i in range(len(batch_decoded[0])):
                validation_decoded.append([])
                for j in range(len(batch_decoded)):
                    validation_decoded[-1].append(batch_decoded[j][i])
            for log_probs in batch_log_prob:
                validation_log_probs.append(log_probs)
        validation_loss /= num_validation_samples
        validation_error /= num_validation_samples
        validation_error_selected /= num_validation_samples
        min_validation_error = min(validation_error_selected, min_validation_error)
        
        errors.append(validation_error)
        error_selecteds.append(validation_error_selected)
        
        print
        print validation_error, validation_error_selected

        plt.figure(1)
        plt.gcf().clear()
        plt.plot(errors)
        plt.plot(error_selecteds)
        plt.pause(0.00001)
        
#        avg_kernel = normalize_kernel([1.0] * (sample_rate // 20))
#        def deprocess(x):
##            x -= np.mean(x)
##            x /= (np.std(x) + 1e-5)
#            x = data.transform.correlate(x, avg_kernel)
##            x *= 0.1
##            x += 0.5
##            x = np.clip(x, 0, 1)
##            x *= 255
###            x = np.transpose(x, (1, 2, 0))
##            x = np.clip(x, 0, 255).astype('uint8')
#            return x
#        test_index = 0
#        num_copies = 10 #20
#        if (epoch+1) % 1 == 0:
##            test_input = np.random.rand((num_classes-1)*num_copies, cnn_window_size, num_channels) * 2 - 1.0
#            test_input = np.zeros(((num_classes-1)*num_copies, cnn_window_size, num_channels))
#            for i in range(300):
#                test_targets = np.zeros(((num_classes-1)*num_copies, num_classes))
#                for c in range(num_classes-1):
#                    test_targets[c*num_copies:(c+1)*num_copies,c] = 1.0
#                gradients_output, logits_output = session.run([vis_gradients, vis_logits],
#                                                              {vis_inputs: test_input, vis_targets: test_targets})
#                test_input += gradients_output[0] * 0.02
#                print logits_output[0]
##                test_input += (np.random.rand(num_classes-1, cnn_window_size, num_channels) - 0.5) * 0.02
#                if (i+1) % 15 == 0 or i == 0:
#                    plt.gcf().clear()
#                    plt.plot(deprocess(np.mean(test_input[test_index*num_copies:(test_index+1)*num_copies,:,:], axis=0)))
#                    plt.pause(0.00001)
#                    print i+1
        
        update_table(epoch, batch, training_loss, training_error,
                     min_validation_error, validation_loss, validation_error, finished=True)
        print
        print 'Training:'
        for i in range(min(5, len(training_decoded))):
#            print train_labels[i], ' => ', training_decoded[i][np.where(training_decoded[i] > -1)]
            print '\t', ' '.join(words[train_labels[i]]), ' =>'
            pairs = zip(training_log_probs[i], range(len(training_decoded[i])))
            pairs = sorted(pairs)[::-1]
            for j in range(len(training_decoded[i])):
                print '\t\t', np.exp(training_log_probs[i][j]), '\t', \
                ' '.join(words[training_decoded[i][j][np.where(training_decoded[i][j] > -1)]]), \
                '\t\t\t', np.exp(pairs[j][0]), '\t', \
                ' '.join(words[training_decoded[i][pairs[j][1]][
                            np.where(training_decoded[i][pairs[j][1]] > -1)]])
        print 'Validation:'
        for i in range(min(5, len(validation_decoded))):
#            print val_labels[i], ' => ', validation_decoded[i][np.where(validation_decoded[i] > -1)]
            print '\t', ' '.join(words[val_labels[i]]), ' =>'
            pairs = zip(validation_log_probs[i], range(len(validation_decoded[i])))
            pairs = sorted(pairs)[::-1]
            for j in range(len(validation_decoded[i])):
                print '\t\t', np.exp(validation_log_probs[i][j]), '\t', \
                ' '.join(words[validation_decoded[i][j][np.where(validation_decoded[i][j] > -1)]]), \
                '\t\t\t', np.exp(pairs[j][0]), '\t', \
                ' '.join(words[validation_decoded[i][pairs[j][1]][
                            np.where(validation_decoded[i][pairs[j][1]] > -1)]])
        
        reprint_header = (epoch+1) % 10 == 0 and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()
