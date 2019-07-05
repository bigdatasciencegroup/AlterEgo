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

def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
def ricker_function(t, sigma):
    return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))
    
def transform_data(sequence_groups, sample_rate=1000):
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
sequence_groups = transform_data(data.words_10_20_sentences_dataset(include_surrounding=False))
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
            
training_sequence_groups, validation_sequence_groups = train_test_split(sequence_groups, range(2, 20, 5))
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
cold water want hungry i
[2, 9, 6, 4, 0]
water tired hot i food
[9, 5, 3, 0, 8]
food hungry need hot tired
[8, 4, 7, 3, 5]
cold hot want water am
[2, 3, 6, 9, 1]
need want tired cold am
[7, 6, 5, 2, 1]
hot food tired am hungry
[3, 8, 5, 1, 4]
hungry water need i want
[4, 9, 7, 0, 6]
water hungry tired need food
[9, 4, 5, 7, 8]
food hot am cold i
[8, 3, 1, 2, 0]
water i am need tired
[9, 0, 1, 7, 5]
want hot cold hungry am
[6, 3, 2, 4, 1]
hungry want am water cold
[4, 6, 1, 9, 2]
i need cold tired food
[0, 7, 2, 5, 8]
want food i hot water
[6, 8, 0, 3, 9]
food cold need am i
[8, 2, 7, 1, 0]
tired hungry food am hot
[5, 4, 8, 1, 3]
need hungry cold want i
[7, 4, 2, 6, 0]
cold food want need water
[2, 8, 6, 7, 9]
am tired water hot hungry
[1, 5, 9, 3, 4]
tired i hungry hot need
[5, 0, 4, 3, 7]

Official label order:
[
	0        'i',
	1        'am',
	2        'cold',
	3        'hot',
	4        'hungry',
	5        'tired',
	6        'want',
	7        'need',
	8        'food',
	9        'water',
]
'''
words = np.array(['i', 'am', 'cold', 'hot', 'hungry', 'tired', 'want', 'need', 'food', 'water'])
label_map = [
    [2, 9, 6, 4, 0],
    [9, 5, 3, 0, 8],
    [8, 4, 7, 3, 5],
    [2, 3, 6, 9, 1],
    [7, 6, 5, 2, 1],
    [3, 8, 5, 1, 4],
    [4, 9, 7, 0, 6],
    [9, 4, 5, 7, 8],
    [8, 3, 1, 2, 0],
    [9, 0, 1, 7, 5],
    [6, 3, 2, 4, 1],
    [4, 6, 1, 9, 2],
    [0, 7, 2, 5, 8],
    [6, 8, 0, 3, 9],
    [8, 2, 7, 1, 0],
    [5, 4, 8, 1, 3],
    [7, 4, 2, 6, 0],
    [2, 8, 6, 7, 9],
    [1, 5, 9, 3, 4],
    [5, 0, 4, 3, 7],
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

num_channels = len(train_sequences[0][0])


learning_rate = 1e-4 #5e-4 #1e-3
dropout_rate = 0.4

num_channels = 8

inputs = tf.placeholder(tf.float32,[None, None, num_channels]) #[batch_size,timestep,features]
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

reshaped = tf.reshape(outputs, [-1, 2 * lstm_hidden_size])

reshaped = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(2 * lstm_hidden_size * 1024))))
reshaped = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(1024 * 1024))))
logits = tf.layers.dense(reshaped, num_classes,
            kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(1024 * num_classes))))


logits = tf.reshape(logits, [batch_size, -1, num_classes])
logits = tf.transpose(logits, [1, 0, 2])

loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=sequence_lengths, time_major=True)
loss = tf.reduce_mean(tf.multiply(loss, weights))

# L2 regularization
#l2 = 0.005 * sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()\
#                          if 'noreg' not in tf_var.name.lower() and 'bias' not in tf_var.name.lower()])
#loss += l2

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

#decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sequence_lengths)
ctc_output = tf.nn.ctc_beam_search_decoder(logits, sequence_lengths, top_paths=10)
decoded, log_prob = ctc_output[0], ctc_output[1]

error = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets, normalize=True))

decoded_selected = tf.sparse_placeholder(tf.int32)
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
            batch_lengths = map(len, batch_sequences)
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(batch_lengths), position=0.0)
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
            batch_lengths = map(len, batch_sequences)
            batch_sequences = data.transform.pad_truncate(batch_sequences, max(batch_lengths), position=0.0)
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
                pairs = sorted(pairs, key=lambda x: x[0])
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

#        plt.figure(1)
#        plt.gcf().clear()
#        plt.plot(errors)
#        plt.plot(error_selecteds)
#        plt.pause(0.00001)

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
