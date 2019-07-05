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

length = 600 # 300,600
    
# Load data and apply transformations
sequence_groups = data.digits_dataset()

sequence_groups = data.transform.default_transform(sequence_groups)

# Split into training and validation data
training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./6)

training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, length)
validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, length)

# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)

# Calculate sample weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
train_weights = class_weights[list(train_labels)]

#train_labels = tf.keras.utils.to_categorical(train_labels)
#val_labels = tf.keras.utils.to_categorical(val_labels)
label_map = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]]
train_labels = np.array(map(lambda i: label_map[i], train_labels))
val_labels = np.array(map(lambda i: label_map[i], val_labels))

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

num_classes = len(np.unique(reduce(lambda a,b: a+b, label_map))) + 1


learning_rate = 1e-4
dropout_rate = 0.4

inputs = tf.placeholder(tf.float32,[None, None, np.shape(train_sequences[0])[1]]) #[batch_size,timestep,features]
targets = tf.sparse_placeholder(tf.int32)
sequence_lengths = tf.placeholder(tf.int32, [None])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)
batch_size = tf.shape(inputs)[0]
max_timesteps = tf.shape(inputs)[1]

lstm_hidden_size = 250
lstm1 = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, 1.0-dropout_rate)
lstm1b = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
dropout1b = tf.nn.rnn_cell.DropoutWrapper(lstm1b, 1.0-dropout_rate)

forward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout1])
backward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout1b])

outputs, states = tf.nn.bidirectional_dynamic_rnn(forward_stack, backward_stack, inputs, dtype=tf.float32)

#outputs = tf.transpose(outputs, [1, 0, 2])
#last_output = outputs[-1]
outputs = tf.convert_to_tensor(outputs)
#outputs = tf.reduce_mean(outputs, axis=1)

#reshaped = tf.reshape(outputs, [-1, lstm_hidden_size])
##reshaped = outputs
##fc1 = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu)
##fc1 = tf.layers.dropout(fc1, dropout_rate, training=training)
##logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax)
##logits = tf.layers.dense(reshaped, num_classes, activation=tf.nn.softmax)
#logits = tf.layers.dense(reshaped, num_classes)


reshaped = tf.reshape(outputs, [-1, lstm_hidden_size])
W = tf.Variable(tf.truncated_normal([lstm_hidden_size, num_classes], stddev = 0.1))
b = tf.Variable(tf.constant(0.0, shape = [num_classes]))
logits = tf.matmul(reshaped, W) + b


logits = tf.reshape(logits, [batch_size, -1, num_classes])
logits = tf.transpose(logits, [1, 0, 2])

#loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))
loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=sequence_lengths, time_major=True)
loss = tf.reduce_mean(tf.multiply(loss, weights))

#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

#opt = tf.train.RMSPropOptimizer(learning_rate)
#gradients = opt.compute_gradients(loss)
#clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
#optimizer = opt.apply_gradients(clipped_gradients)

decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sequence_lengths)
error = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets, normalize=True))


num_epochs = 200 #200
batch_size = 500 #50

num_training_samples = len(train_sequences)
num_batches = max(1, int(num_training_samples / batch_size))
start_time = None
last_time = None

# Table display
progress_bar_size = 20
layout = [
    dict(name='Ep.', width=len(str(num_epochs)), align='center'),
    dict(name='Batch', width=2*len(str(num_batches))+1, suffix='/'+str(num_batches), align='center'),
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

def update_table(epoch, batch, training_loss, training_error, min_validation_error,
                 validation_loss=None, validation_error=None):
    global last_time
    progress = int(math.ceil(progress_bar_size * float(batch) / num_batches))
    progress_string = '[' + '#' * progress + ' ' * (progress_bar_size - progress) + ']'
    now = time.time()
    start_elapsed = now - start_time
    epoch_elapsed = now - last_time
    batch_time_estimate = epoch_elapsed / batch if batch else 0.0
    eta_string = batch_time_estimate * (num_batches - batch) or '--'
    if validation_loss:
        last_time = now
        progress_string = time.strftime("%I:%M:%S %p",time.localtime())+'; '+str(int(start_elapsed*10)/10.)+'s'
        eta_string = epoch_elapsed
    table.update(epoch + 1, batch + 1,
                 progress_string, eta_string, '',
                 training_loss or '--', training_error or '--', '',
                 validation_loss or '--', validation_error or '--', '',
                 min_validation_error if validation_error else '--')
            
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
        for batch in range(num_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            actual_batch_size = len(indices)
            if batch == num_batches - 1:
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
                        
            training_loss += batch_loss * actual_batch_size
            training_error += batch_error * actual_batch_size
            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded[0], default_value=-1).eval()
            for seq in batch_decoded:
                training_decoded.append(seq)
            
        training_loss /= num_training_samples
        training_error /= num_training_samples
                
        validation_loss = 0.0
        validation_error = 0.0
        validation_decoded = []
        for b in range(int(max(1, len(val_sequences) / batch_size))):         
            indices = range(b * batch_size, (b + 1) * batch_size)
            if b == int(max(1, len(val_sequences) / batch_size)) - 1:
                indices = range(b * batch_size, len(val_sequences))
            batch_sequences = val_sequences[indices]
            batch_labels = val_labels[indices]
            batch_weights = np.ones(len(indices))
            batch_lengths = map(len, batch_sequences)
            sparse_batch_labels = sparsify(batch_labels)
            
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(batch_lengths), position=0.0)
            
            validation_feed = {inputs: batch_sequences, targets: sparse_batch_labels,
                               sequence_lengths: batch_lengths, weights: batch_weights, training: False}
            batch_output = session.run([loss, error, decoded, log_prob], validation_feed)
            batch_loss, batch_error, batch_decoded, batch_log_prob = batch_output
            validation_loss += batch_loss * len(indices)
            validation_error += batch_error * len(indices)
            batch_decoded = tf.sparse_tensor_to_dense(batch_decoded[0], default_value=-1).eval()
            for seq in batch_decoded:
                validation_decoded.append(seq)
        validation_loss /= len(val_sequences)
        validation_error /= len(val_sequences)
        min_validation_error = min(validation_error, min_validation_error)
                
        update_table(epoch, batch, training_loss, training_error,
                     min_validation_error, validation_loss, validation_error)
        print
        print 'Training:'
        for i in range(min(10, len(training_decoded))):
            print train_labels[i], ' => ', training_decoded[i][np.where(training_decoded[i] > -1)]
        print 'Validation:'
        for i in range(min(10, len(validation_decoded))):
            print val_labels[i], ' => ', validation_decoded[i][np.where(validation_decoded[i] > -1)]
        
        reprint_header = (epoch+1) % 10 == 0 and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()
