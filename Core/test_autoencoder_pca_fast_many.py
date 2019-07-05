import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from display_utils import DynamicConsoleTable
import time
import math
import os

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


sequence_groups = transform_data(
    data.words_10_20_sentences_dataset(include_surrounding=False, channels=range(0, 8)))

#sequence_groups = transform_data(
#    data.combine([
#            data.digits_session_dependence_1_dataset(channels=range(1, 8)),
#            data.digits_session_dependence_2_dataset(channels=range(1, 8)),
#            data.digits_session_dependence_3_dataset(channels=range(1, 8)),
#        ]))

#sequence_groups = transform_data(data.digits_session_4_dataset(channels=range(1, 8)), sample_rate=250)

#sequence_groups = transform_data(
#    data.combine([
#            data.digits_session_5_dataset(channels=range(4, 8)),
#            data.digits_session_6_dataset(channels=range(4, 8)),
#            data.digits_session_7_dataset(channels=range(4, 8)),
#        ]), sample_rate=250)

training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./6)

train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)

num_channels = len(train_sequences[0][0])

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

lens = map(len, train_sequences)
print min(lens), np.mean(lens), max(lens)


def window_split(sequence, window, stride=None, step=1):
    if stride is None: stride = window
    return [sequence[i:i+window:step] for i in range(0, len(sequence)-window+1, stride)]
#def center_window(sequence, window):
#    start = (len(sequence)-window)//2
#    return np.array(np.concatenate(sequence[start:start+window]))
#def window_split_and_spans(sequence, window, stride=None, step=1):
#    if stride is None: stride = window
#    spans = map(lambda i: (i, i+window), range(0, len(sequence)-window+1, stride))
#    return np.array([np.concatenate(sequence[i:j:step]) for i, j in spans]), spans
    

window_size = 250 #250, 300 1000, 2000
stride = 40 # 250, 125

train_sequences = np.array(sum(map(lambda seq: window_split(seq, window_size, stride, 1), train_sequences), []))
val_sequences = np.array(sum(map(lambda seq: window_split(seq, window_size, stride, 1), val_sequences), []))

print np.shape(train_sequences)
print np.shape(val_sequences)


#### Model
learning_rate = 1e-4
dropout_rate = 0.4

latent_size = 32

inputs = tf.placeholder(tf.float32,[None, window_size, num_channels])
targets = tf.placeholder(tf.float32,[None, window_size, num_channels])
training = tf.placeholder(tf.bool)

conv1 = tf.layers.conv1d(inputs, 400, 12, activation=tf.nn.relu, padding='valid') # 50
conv1 = tf.layers.max_pooling1d(conv1, 2, strides=2)
conv2 = tf.layers.conv1d(conv1, 400, 6, activation=tf.nn.relu, padding='valid') # 25
conv2 = tf.layers.max_pooling1d(conv2, 2, strides=2)
conv3 = tf.layers.conv1d(conv2, 400, 3, activation=tf.nn.relu, padding='valid') # 12
conv3 = tf.layers.max_pooling1d(conv3, 2, strides=2)
conv4 = tf.layers.conv1d(conv3, 400, 3, activation=tf.nn.relu, padding='valid') # 12
conv4 = tf.layers.max_pooling1d(conv4, 2, strides=2)
conv5 = tf.layers.conv1d(conv4, 400, 3, activation=tf.nn.relu, padding='valid') # 12
conv5 = tf.layers.max_pooling1d(conv5, 2, strides=2)
dropout = tf.layers.dropout(conv5, dropout_rate, training=training)
reshaped = tf.reshape(dropout, [-1, np.prod(dropout.shape[1:])])
fc1 = tf.layers.dense(reshaped, latent_size, activation=tf.nn.relu) # 125
fc2 = tf.layers.dense(fc1, 250, activation=tf.nn.relu)
#fc2 = tf.layers.dense(reshaped, 250, activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2, 500, activation=tf.nn.relu)
outputs = tf.layers.dense(fc3, window_size * num_channels)
outputs = tf.reshape(outputs, [-1, window_size, num_channels])

loss = tf.reduce_mean(tf.losses.mean_squared_error(targets, outputs))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


num_epochs = 1
batch_size = 50 #50

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
    dict(name='', width=0, align='center'),
    dict(name='Val Loss', width=8, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Min Val Loss', width=7, align='center'),
]
since_training = 0
def update_table(epoch, batch, training_loss, min_validation_loss,
                 validation_loss=None, finished=False):
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
                 training_loss or '--', '',
                 validation_loss or '--', '',
                 min_validation_loss if finished else '--')

saver = tf.train.Saver()
with tf.Session() as session:
    tf.global_variables_initializer().run()
    saver.restore(session, 'autoencoder_' + str(window_size) + '_' + str(latent_size) + '_model.ckpt')
    
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    start_time = time.time()
    last_time = start_time
    
    # Training/validation loop
    min_validation_loss = float('inf')
    permutation = np.random.permutation(num_validation_samples)
    val_sequences = val_sequences[permutation]
    for epoch in range(num_epochs):
        # Training
        training_loss = 0.0
        permutation = np.random.permutation(num_training_samples)
        train_sequences = train_sequences[permutation]
        encodings = None
        for batch in range(num_training_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_training_batches - 1:
                indices = range(batch * batch_size, num_training_samples)
            batch_sequences = train_sequences[indices]
            
            update_table(epoch, batch, training_loss / (batch_size * max(1, batch)), min_validation_loss)
                        
            training_feed = {inputs: batch_sequences, targets: batch_sequences, training: True}
            batch_loss, batch_encodings = session.run([loss, fc1], training_feed)
                        
            training_loss += batch_loss * len(indices)
            
            encodings = batch_encodings if encodings is None else np.concatenate([encodings, batch_encodings], axis=0)
            
        training_loss /= num_training_samples
        
        # Validation
        validation_loss = 0.0
        output = None
        for batch in range(num_validation_batches):         
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_validation_batches - 1:
                indices = range(batch * batch_size, num_validation_samples)
            batch_sequences = val_sequences[indices]
    
            update_table(epoch, batch, training_loss, min_validation_loss,
                         validation_loss / (batch_size * max(1, batch)))
            
            validation_feed = {inputs: batch_sequences, targets: batch_sequences, training: False}
            batch_loss, batch_output, batch_encodings = session.run([loss, outputs, fc1], validation_feed)
            
            validation_loss += batch_loss * len(indices)
            
            output = batch_output if output is None else np.concatenate([output, batch_output], axis=0)
            encodings = batch_encodings if encodings is None else np.concatenate([encodings, batch_encodings], axis=0)
            
        validation_loss /= num_validation_samples
#        if validation_loss < min_validation_loss:
#            save_path = saver.save(session, os.path.join(
#                    abs_path, 'autoencoder_' + str(window_size) + '_model.ckpt'))
#            print '\n\nModel saved:', save_path, '\n'
        min_validation_loss = min(validation_loss, min_validation_loss)
                
        update_table(epoch, batch, training_loss, min_validation_loss, validation_loss, finished=True)
        
        print
        print list(encodings[0])
        num_components = 32
        mu = np.mean(encodings, axis=0)
        pca = PCA(n_components=num_components)
        transformed = pca.fit_transform(encodings)
        eigenvalues = pca.explained_variance_
        print list(transformed[0])
        reconstructed = np.dot(transformed[:,:num_components], pca.components_[:num_components,:])
        reconstructed += mu
        print list(reconstructed[0])
        
#        original_output = session.run(outputs, {fc1: [encodings[0]]})
#        reconstructed_output = session.run(outputs, {fc1: [reconstructed[0]]})
#        plt.plot(original_output[0])
#        plt.figure()
#        plt.plot(reconstructed_output[0])
#        plt.show()
        
        num_shown = 32
        num_cols = 8
        downsample_factor = 1
        fig, axes = plt.subplots(int(math.ceil(float(num_shown) / num_cols)), num_cols, figsize=(18, 8))
        plt.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.93, wspace=0.35, hspace=0.35)
        plt.suptitle('Effect of Each Principal Component on Autoencoder Output', y=0.995, fontweight='bold')
        colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
        lines = [[axes[i//num_cols][i%num_cols].plot([],[], '-', c=colors[c])[0] for c in range(8)] for i in range(num_shown)]
        for i in range(num_shown):
            axes[i//num_cols][i%num_cols].axis([0, window_size//downsample_factor, -125, 125])
            axes[i//num_cols][i%num_cols].set_title('PC' + str(i+1) + ' (' + str(eigenvalues[i]) + ')')
        flat = pca.transform(session.run(fc1, {inputs: [[[0.0] * num_channels for _ in range(window_size)]], training: False}))[0]
        mean = pca.transform([mu])[0]
        xs = list(range(window_size//downsample_factor))
        test_count = 0
        last_value = 0
        def update(i):
            global test_count
            global last_value
#            test_encoding = np.zeros(np.shape(encodings[0]))
#            test_encoding = np.array(transformed[0])
#            test_encodings = [np.array(flat) for _ in range(num_shown)]
            test_encodings = [np.array(mean) for _ in range(num_shown)]
#            test_encodings = [np.array(mu) for _ in range(num_shown)]

            if last_value == 500:
                plt.savefig('figure_autoencoder_principal_components_pos.png')
            if last_value == -500:
                plt.savefig('figure_autoencoder_principal_components_neg.png')

            value = 500 * np.sin(10 * test_count * np.pi / 180.0)
            last_value = value
            for j in range(num_shown):
                test_encodings[j][j] += value
            print value
            test_count += 1
            test_reconstructeds = np.dot(np.array(test_encodings)[:,:num_components], pca.components_[:num_components,:])
            test_reconstructeds += mu
#            test_reconstructeds = test_encodings
            reconstructed_outputs = session.run(outputs, {fc1: test_reconstructeds})
            if downsample_factor > 1:
                reconstructed_outputs = data.transform.downsample(reconstructed_outputs, downsample_factor)
            for j in range(num_shown):
                for c in range(8):
                    lines[j][c].set_data(xs, reconstructed_outputs[j][:,c])
    #                plt.plot(reconstructed_output[0][:,c], c=colors[c])
            return sum(lines, [])
            
        line_ani = animation.FuncAnimation(fig, update, interval=0, blit=True)
#        update(0)
        plt.show()
        
#        if (epoch + 1) % 1 == 0:
#            save_path = saver.save(session, os.path.join(
#                    abs_path, 'autoencoder_' + str(window_size) + '_model.ckpt'))
#            print ' Model saved!', #save_path,
            
#        colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
#        if (epoch + 1) % 1 == 0: # 10
##            shown = range(20)
#            shown = [7, 9, 10]
#            fig, axes = plt.subplots(len(shown), 2, figsize=(12, 4))
#            plt.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.92, wspace=0.10, hspace=0.35)
#            axes[0][0].set_title('Original Signal')
#            axes[0][1].set_title('Autoencoder Output')
#            for i in range(len(shown)):
#                for c in range(8):
#                    axes[i][0].plot(val_sequences[shown[i]][:,c], c=colors[c])
#                    axes[i][1].plot(output[shown[i]][:,c], c=colors[c])
##            plt.savefig('figure_autoencoder_validation.png')
#            plt.show()
        
        reprint_header = (epoch+1) % 10 == 0 and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()

