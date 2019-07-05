import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from display_utils import DynamicConsoleTable
import math
import time
import fnmatch
import os

import data

def load_data(data_dir, labels_dir):
    print 'Loading data from', data_dir
    matches = []
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, '*.npy'):
            matches.append(os.path.join(root, filename))
    sequences = np.array(map(lambda filepath: list(np.transpose(np.load(filepath))), sorted(matches)))
    print 'Loading labels from', labels_dir
    matches = []
    for root, dirnames, filenames in os.walk(labels_dir):
        for filename in fnmatch.filter(filenames, '*.npy'):
            matches.append(os.path.join(root, filename))
#    print matches
    labels = np.array(map(lambda filepath: list(np.load(filepath)), sorted(matches)))
    return sequences, labels

train_sequences, train_labels = load_data('timit/tensorflow_CTC_example/train_data/data',
                                          'timit/tensorflow_CTC_example/train_data/labels')
val_sequences, val_labels = load_data('timit/tensorflow_CTC_example/test_data/data',
                                      'timit/tensorflow_CTC_example/test_data/labels')

#train_sequences, train_labels = load_data('timit/tensorflow_CTC_example/sample_data/mfcc',
#                                          'timit/tensorflow_CTC_example/sample_data/char_y')
#val_sequences, val_labels = load_data('timit/tensorflow_CTC_example/sample_data/mfcc',
#                                      'timit/tensorflow_CTC_example/sample_data/char_y')

#train_sequences, train_labels = load_data('timit/tensorflow_CTC_example/medium_train_data/data',
#                                          'timit/tensorflow_CTC_example/medium_train_data/labels')
#val_sequences, val_labels = load_data('timit/tensorflow_CTC_example/medium_test_data/data',
#                                      'timit/tensorflow_CTC_example/medium_test_data/labels')

#train_sequences, train_labels = load_data('timit/tensorflow_CTC_example/small_train_data/data',
#                                          'timit/tensorflow_CTC_example/small_train_data/labels')
#val_sequences, val_labels = load_data('timit/tensorflow_CTC_example/small_test_data/data',
#                                      'timit/tensorflow_CTC_example/small_test_data/labels')


#train_sequences = train_sequences[:500]
#train_labels = train_labels[:500]
#val_sequences = val_sequences[:250]
#val_labels = val_labels[:250]


print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

train_weights = np.ones(len(train_labels))
num_classes = len(np.unique(sum(map(list, train_labels), []))) + 1
#num_classes = 28
#num_classes = 61
print num_classes


learning_rate = 5e-3 #1e-3
dropout_rate = 0.4

inputs = tf.placeholder(tf.float32,[None, None, 20]) #[batch_size,timestep,features]
targets = tf.sparse_placeholder(tf.int32)
sequence_lengths = tf.placeholder(tf.int32, [None])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)
batch_size = tf.shape(inputs)[0]
max_timesteps = tf.shape(inputs)[1]

lstm_hidden_size = 256
lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, 1.0-dropout_rate)
lstm1b = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
dropout1b = tf.nn.rnn_cell.DropoutWrapper(lstm1b, 1.0-dropout_rate)
lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, 1.0-dropout_rate)
lstm2b = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
dropout2b = tf.nn.rnn_cell.DropoutWrapper(lstm2b, 1.0-dropout_rate)

forward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout2])
backward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout2b])

outputs, states = tf.nn.bidirectional_dynamic_rnn(forward_stack, backward_stack, inputs, dtype=tf.float32)
outputs = tf.concat(outputs, 2)

#weightsOutH1 = tf.Variable(tf.truncated_normal([2, lstm_hidden_size], stddev=np.sqrt(2.0 / (2*lstm_hidden_size))))
#biasesOutH1 = tf.Variable(tf.zeros([lstm_hidden_size]))
#weightsClasses = tf.Variable(tf.truncated_normal([lstm_hidden_size, num_classes],
#                                                 stddev=np.sqrt(2.0 / (lstm_hidden_size*num_classes))))
#biasesClasses = tf.Variable(tf.zeros([num_classes]))
#
#outputs = [tf.reshape(t, [batch_size, 2, lstm_hidden_size]) for t in outputs]
#outputs = [tf.reduce_sum(tf.multiply(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in outputs]
#
#logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outputs]
#logits = tf.stack(logits)
#
#print logits

#reshaped = tf.reshape(outputs, [-1, lstm_hidden_size])
reshaped = tf.reshape(outputs, [-1, 2 * lstm_hidden_size])

#W1 = tf.Variable(tf.truncated_normal([2 * lstm_hidden_size, 1024],
#                                     stddev=np.sqrt(2.0/(2 * lstm_hidden_size * 1024))))
#b1 = tf.Variable(tf.constant(0.0, shape=[1024]))
#W2 = tf.Variable(tf.truncated_normal([1024, num_classes],
#                                     stddev=np.sqrt(2.0/(1024 * num_classes))))
#b2 = tf.Variable(tf.constant(0.0, shape=[num_classes]))
#reshaped = tf.matmul(reshaped, W1) + b1
#logits = tf.matmul(reshaped, W2) + b2

reshaped = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(2 * lstm_hidden_size * 1024))))
reshaped = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(1024 * 1024))))

reshaped = tf.layers.dropout(reshaped, rate=dropout_rate, training=training)

logits = tf.layers.dense(reshaped, num_classes,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(1024 * num_classes))))


logits = tf.reshape(logits, [batch_size, -1, num_classes])
logits = tf.transpose(logits, [1, 0, 2])

#loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))
loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=sequence_lengths, time_major=True)
loss = tf.reduce_mean(tf.multiply(loss, weights))

# L2 regularization
l2 = 0.005 * sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()\
                          if 'noreg' not in tf_var.name.lower() and 'bias' not in tf_var.name.lower()])
loss += l2

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
            batch_labels = train_labels[indices]
            batch_weights = train_weights[indices]
            batch_lengths = map(len, batch_sequences)
            sparse_batch_labels = sparsify(batch_labels)
            
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(batch_lengths), position=0.0)
            
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
            batch_labels = val_labels[indices]
            batch_weights = np.ones(len(indices))
            batch_lengths = map(len, batch_sequences)
            sparse_batch_labels = sparsify(batch_labels)
            
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(batch_lengths), position=0.0)
            
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
