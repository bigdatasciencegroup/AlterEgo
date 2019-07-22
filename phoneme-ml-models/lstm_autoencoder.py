'''
Latent dim = 256
Val split = 0.1
Data = train_sequences/3, center slice
Batch size = 30
Epochs = 300
value = 1e-10
Training loss: 394.71
Val loss: 372.12
'''

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional, CuDNNLSTM
from tensorflow.python.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K


import matplotlib
import time
import json
import math
import os.path

# Local imports
import config
import data
from display_utils import DynamicConsoleTable


def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))


def ricker_function(t, sigma):
    return 2. / (np.sqrt(3 * sigma) * np.pi ** 0.25) * (1. - (float(t) / sigma) ** 2) * np.exp(
        -(float(t) ** 2) / (2 * sigma ** 2))


def ricker_wavelet(n, sigma):
    return np.array(map(lambda x: ricker_function(x, sigma), range(-n // 2, n // 2 + 1)))


def transform_data(sequence_groups, sample_rate=250):
    #### Apply DC offset and drift correction
    drift_low_freq = 0.5  # 0.5
    sequence_groups = data.transform.subtract_initial(sequence_groups)
    sequence_groups = data.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    sequence_groups = data.transform.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq
    notch_freq = 60
    num_times = 3  # pretty much just the filter order
    freqs = map(int, map(round, np.arange(1, sample_rate / (2. * notch_freq)) * notch_freq))
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = data.transform.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    # sequence_groups = data.transform.normalize_std(sequence_groups)

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
    ricker_subtraction_multiplier = 2.0
    sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

    #### Apply sine wavelet kernel
    # period = int(sample_rate)
    # sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
    # sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5  # 0.5
    high_freq = 8  # 8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

    #### Apply hard bandpassing
    # sequence_groups = data.transform.fft(sequence_groups)
    # sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    # sequence_groups = np.real(data.transform.ifft(sequence_groups))
    #
    return sequence_groups

with open(config.data_maps, 'r') as f:
    input_data = json.load(f)

training_files = []
test_files = []
for data_file in input_data:
    if data_file['type'] == 'phonemes_utkarsh':
        if 'train' in data_file['filename']:
            train_file = data.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
                                       sample_rate=config.sample_rate, surrounding=config.surrounding, exclude=set([]),
                                       num_classes=config.num_classes)
            training_files.append(train_file)
        if 'test' in data_file['filename']:
            test_file = data.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
                                       sample_rate=config.sample_rate, surrounding=config.surrounding,
                                       exclude=set([]), num_classes=config.num_classes)
            test_files.append(test_file)

training_sequence_groups = data.combine(training_files)
test_sequence_groups = data.combine(test_files)

print("Training sequences:")
print(len(training_sequence_groups), " sequences")
lens = map(len, data.get_inputs(training_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

print("Validation sequences:")
print(len(test_sequence_groups), "sequences")
lens = map(len, data.get_inputs(test_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
test_sequences, test_labels = data.get_inputs(test_sequence_groups)

train_sequences = transform_data(train_sequences)
test_sequences = transform_data(test_sequences)


label_map = config.phoneme_label_map
print("Label map:", len(label_map))
num_classes = len(np.unique(reduce(lambda a,b: a+b, label_map))) + 2 #(for start and end symbols)
start_symbol = num_classes - 2
end_symbol = num_classes - 1


label_map = map(lambda label_seq: [start_symbol] + label_seq + [end_symbol], label_map)
label_map = map(lambda label_seq: tf.keras.utils.to_categorical(label_seq, num_classes=num_classes), label_map)

train_labels = np.array(map(lambda i: label_map[i], train_labels))
test_labels = np.array(map(lambda i: label_map[i], test_labels))

max_input_length = max(map(len, train_sequences) + map(len, test_sequences))
max_labels_length = max(map(len, train_labels) + map(len, test_labels))

train_sequences = data.transform.pad_truncate(train_sequences, max_input_length, position=0.0, value=1e-10)
test_sequences = data.transform.pad_truncate(test_sequences, max_input_length, position=0.0, value=1e-10)
train_labels = data.transform.pad_truncate(train_labels, max_labels_length, position=0.0, value=0)
test_labels = data.transform.pad_truncate(test_labels, max_labels_length, position=0.0, value=0)

print("Number of classes: ", num_classes)
print("Number of samples: ", np.shape(train_sequences)[0] + np.shape(test_sequences)[0])

_, train_sequences, _ = np.split(train_sequences, 3)

# Hyperparameters
learning_rate = 0.001 # 0.001
epochs = 300
batch_size = 30 # 20 # 50
latent_dim = 256

fold = 1

# Result logging
timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
log_name = "{}_e{}_b{}_phon_bidir_utkarsh_CV".format(timeString, epochs, batch_size)
result_file = open(log_name + ".txt", "w")
result_file.write('# HYPERPARAMETERS:\nepochs:{}\nbatch size:{}\nlatent dim:{}\nlearning rate:{}\n'.format(epochs, batch_size, latent_dim, learning_rate))
result_file.write('epoch, training_loss, training_acc, max_validation_accuracy, val_loss, validation_accuracy\n')

(_,x,_) = train_sequences.shape

# Autoencoder
model = Sequential()
timesteps = x # 1572
n_features = len(config.channels)
# model.add(LSTM(256, input_shape=(timesteps, n_features), return_sequences=True)) # try different activation functions, default is tanh
# model.add(LSTM(128))
model.add(LSTM(latent_dim, input_shape=(timesteps, n_features), activation='relu', recurrent_activation='relu')) # 1 x latent_dim
model.add(RepeatVector(timesteps)) # timesteps x latent_dim
# model.add(LSTM(128, return_sequences=True))
# model.add(LSTM(256, return_sequences=True))
model.add(LSTM(latent_dim, return_sequences=True, activation='relu', recurrent_activation='relu')) # timesteps x latent_dim
model.add(TimeDistributed(Dense(n_features))) # Creates a Dense layer of size latemt_dim and duplicates it n_features times 
model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mse')

# # Table display
# num_training_samples = len(train_sequences)
# num_validation_samples = len(test_sequences)
# num_training_batches = int(math.ceil(float(num_training_samples) / batch_size))
# num_validation_batches = int(math.ceil(float(num_validation_samples) / batch_size))
# start_time = None
# last_time = None
# progress_bar_size = 20
# max_batches = max(num_training_batches, num_validation_batches)
# layout = [
#     dict(name='Ep.', width=len(str(epochs)), align='center'),
#     dict(name='Batch', width=2 * len(str(max_batches)) + 1, align='center'),
#     dict(name='Progress/Timestamp', width=progress_bar_size + 2, align='center'),
#     dict(name='ETA/Elapsed', width=7, suffix='s', align='center'),
#     dict(name='', width=0, align='center'),
#     dict(name='Train Loss', width=8, align='center'),
#     dict(name='Train Acc', width=7, align='center'),
#     dict(name='', width=0, align='center'),
#     dict(name='Val Loss', width=8, align='center'),
#     dict(name='Val Acc', width=7, align='center'),
#     dict(name='', width=0, align='center'),
#     dict(name='Max Val Acc', width=7, align='center'),
# ]

# table = DynamicConsoleTable(layout)

# training_losses = []
# training_accuracies = []
# validation_losses = []
# validation_accuracies = []

# since_training = 0

# def update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
#                  validation_loss=None, validation_accuracy=None, finished=False):
#     global last_time
#     global since_training
#     num_batches = num_training_batches
#     progress = int(math.ceil(progress_bar_size * float(batch) / num_batches))
#     status = ' Training' if validation_loss is None else ' Validating'
#     status = status[:max(0, progress_bar_size - progress)]
#     progress_string = '[' + '#' * progress + status + ' ' * (progress_bar_size - progress - len(status)) + ']'
#     now = time.time()
#     start_elapsed = now - start_time
#     if validation_loss is None:
#         epoch_elapsed = now - last_time
#         since_training = now
#     else:
#         epoch_elapsed = now - since_training
#     batch_time_estimate = epoch_elapsed / batch if batch else 0.0
#     eta_string = batch_time_estimate * (num_batches - batch) or '--'
#     if finished:
#         epoch_elapsed = now - last_time
#         last_time = now
#         progress_string = time.strftime("%I:%M:%S %p", time.localtime()) + '; ' + str(
#             int(start_elapsed * 10) / 10.) + 's'
#         eta_string = epoch_elapsed
#         training_losses.append(training_loss)
#         training_accuracies.append(training_accuracy)
#         validation_losses.append(validation_loss)
#         validation_accuracies.append(validation_accuracy)
#     table.update(epoch + 1, str(batch + 1) + '/' + str(num_batches),
#                  progress_string, eta_string, '',
#                  training_loss or '--', training_accuracy or '--', '',
#                  validation_loss or '--', validation_accuracy or '--', '',
#                  max_validation_accuracy if finished else '--')


# class TrainingCallbacks(Callback):
#     def __init__(self):
#         super(TrainingCallbacks, self).__init__()
#         self.targets = None
#         self.outputs = None
#         self.batch_targets = tf.Variable(0., validate_shape=False)
#         self.batch_outputs = tf.Variable(0., validate_shape=False)

#     def on_train_begin(self, logs={}):
#         global start_time
#         global last_time
#         table.print_header()
#         start_time = time.time()
#         last_time = start_time
#         self.max_validation_accuracy = 0.0

#     def on_epoch_begin(self, epoch, logs={}):
#         self.targets = None
#         self.outputs = None
#         self.training_loss = 0.0
#         self.training_accuracy = 0.0
#         self.epoch = epoch

#     def on_batch_begin(self, batch, logs={}):
#         batch_size = logs['size']
#         self.batch = batch
#         update_table(self.epoch, self.batch, self.training_loss / (batch_size * max(1, batch)),
#                      self.training_accuracy / (batch_size * max(1, batch)), self.max_validation_accuracy)

#     def on_batch_end(self, batch, logs={}):
#         batch_targets = K.eval(self.batch_targets)
#         batch_outputs = K.eval(self.batch_outputs)
#         self.targets = batch_targets if self.targets is None else np.concatenate([self.targets, batch_targets], axis=0)
#         self.outputs = batch_outputs if self.outputs is None else np.concatenate([self.outputs, batch_outputs], axis=0)
#         batch_size = logs['size']

#         self.training_loss += logs['loss'] * batch_size
#         self.training_accuracy += logs['acc'] * batch_size

#     def on_epoch_end(self, epoch, logs={}):
#         self.training_loss /= num_training_samples
#         self.training_accuracy /= num_training_samples

#         validation_accuracy = logs['val_acc']
#         self.max_validation_accuracy = max(validation_accuracy, self.max_validation_accuracy)

#         update_table(self.epoch, self.batch, self.training_loss, self.training_accuracy,
#                      self.max_validation_accuracy, logs['val_loss'], validation_accuracy, finished=True)

#         reprint_header = (self.epoch + 1) % 10 == 0 and self.epoch < config.num_epochs - 1
#         table.finalize(divider=not reprint_header)
#         if reprint_header:
#             table.print_header()

#         result_file.write(
#             "{}, {}, {}, {}, {}, {}, {}\n".format(self.epoch, self.batch, self.training_loss, self.training_accuracy,
#                                                   self.max_validation_accuracy, logs['val_loss'], validation_accuracy))

#     def on_train_end(self, logs={}):
#         pass



# training_callbacks = TrainingCallbacks()


# fetches = [tf.assign(training_callbacks.batch_targets, model.targets[0], validate_shape=False),
#            tf.assign(training_callbacks.batch_outputs, model.outputs[0], validate_shape=False)]
# model._function_kwargs = {'fetches': fetches}

# localtime = time.localtime()
# timeString = time.strftime("%Y%m%d-%H%M%S", localtime)


tensorboard = TensorBoard(log_dir="logs_AE/{}_f{}".format(log_name, str(fold)), histogram_freq=0, write_graph=True, write_images=True)


# model.fit(train_sequences, train_sequences, validation_split=0.1, epochs=epochs, batch_size=batch_size, 
# 	callbacks=[tensorboard, training_callbacks], verbose=0)

model_history = model.fit(train_sequences, train_sequences, validation_split=0.1,
	callbacks = [tensorboard], epochs=epochs, batch_size=batch_size, verbose=2).history

print(model.summary())
result_file.write(model.summary())


