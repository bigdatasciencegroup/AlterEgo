from __future__ import print_function
import numpy as np
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform
import config
import json
import ml_metrics
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, CuDNNLSTM, LSTM, Dense, Concatenate, Bidirectional
from tensorflow.python.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
import matplotlib
from display_utils import DynamicConsoleTable
import math
import time
import data_proc
import config
import json

matplotlib.use('TkAgg')
os.environ['KERAS_BACKEND'] = 'tensorflow'

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
    sequence_groups = data_proc.transform.subtract_initial(sequence_groups)
    sequence_groups = data_proc.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    sequence_groups = data_proc.transform.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq
    notch_freq = 60
    num_times = 3  # pretty much just the filter order
    freqs = map(int, map(round, np.arange(1, sample_rate / (2. * notch_freq)) * notch_freq))
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = data_proc.transform.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    # sequence_groups = data.transform.normalize_std(sequence_groups)

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = data_proc.transform.correlate(sequence_groups, ricker_kernel)
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
    sequence_groups = data_proc.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

    #### Apply hard bandpassing
    # sequence_groups = data.transform.fft(sequence_groups)
    # sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    # sequence_groups = np.real(data.transform.ifft(sequence_groups))
    #
    return sequence_groups


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

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
        sequence_groups = data_proc.transform.subtract_initial(sequence_groups)
        sequence_groups = data_proc.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
        sequence_groups = data_proc.transform.subtract_mean(sequence_groups)

        #### Apply notch filters at multiples of notch_freq
        notch_freq = 60
        num_times = 3  # pretty much just the filter order
        freqs = map(int, map(round, np.arange(1, sample_rate / (2. * notch_freq)) * notch_freq))
        for _ in range(num_times):
            for f in reversed(freqs):
                sequence_groups = data_proc.transform.notch_filter(sequence_groups, f, sample_rate)

        #### Apply standard deviation normalization
        # sequence_groups = data.transform.normalize_std(sequence_groups)

        #### Apply ricker wavelet subtraction
        ricker_width = 35 * sample_rate // 250
        ricker_sigma = 4.0 * sample_rate / 250
        ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
        ricker_convolved = data_proc.transform.correlate(sequence_groups, ricker_kernel)
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
        sequence_groups = data_proc.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

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
                train_file = data_proc.process_scrambled(data_file['labels'], [config.file_path + data_file['filename']],
                                                    channels=config.channels,
                                                    sample_rate=config.sample_rate, surrounding=config.surrounding,
                                                    exclude=set([]),
                                                    num_classes=config.num_classes)
                training_files.append(train_file)
            if 'test' in data_file['filename']:
                test_file = data_proc.process_scrambled(data_file['labels'], [config.file_path + data_file['filename']],
                                                   channels=config.channels,
                                                   sample_rate=config.sample_rate, surrounding=config.surrounding,
                                                   exclude=set([]), num_classes=config.num_classes)
                test_files.append(test_file)

    training_sequence_groups = data_proc.combine(training_files)
    test_sequence_groups = data_proc.combine(test_files)

    print("Training sequences:")
    print(len(training_sequence_groups), " sequences")
    lens = map(len, data_proc.get_inputs(training_sequence_groups)[0])
    #print min(lens), np.mean(lens), max(lens)

    print("Validation sequences:")
    print(len(test_sequence_groups), "sequences")
    lens = map(len, data_proc.get_inputs(test_sequence_groups)[0])
    #print min(lens), np.mean(lens), max(lens)

    # Format into sequences and labels
    train_sequences, train_labels = data_proc.get_inputs(training_sequence_groups)
    test_sequences, test_labels = data_proc.get_inputs(test_sequence_groups)

    train_sequences = transform_data(train_sequences)
    test_sequences = transform_data(test_sequences)

    label_map = config.phoneme_label_map
    print("Label map:", len(label_map))
    num_classes = len(np.unique(reduce(lambda a, b: a + b, label_map))) + 2  # (for start and end symbols)
    start_symbol = num_classes - 2
    end_symbol = num_classes - 1

    label_map = map(lambda label_seq: [start_symbol] + label_seq + [end_symbol], label_map)
    label_map = map(lambda label_seq: tf.keras.utils.to_categorical(label_seq, num_classes=num_classes), label_map)

    train_labels = np.array(map(lambda i: label_map[i], train_labels))
    test_labels = np.array(map(lambda i: label_map[i], test_labels))
    print(train_labels.shape)

    max_input_length = max(map(len, train_sequences) + map(len, test_sequences))
    max_labels_length = max(map(len, train_labels) + map(len, test_labels))
    print(max_input_length)

    x_train = data_proc.transform.pad_truncate(train_sequences, max_input_length, position=0.0, value=-1e8)
    y_train = data_proc.transform.pad_truncate(test_sequences, max_input_length, position=0.0, value=-1e8)
    x_test = data_proc.transform.pad_truncate(train_labels, max_labels_length, position=0.0, value=0)
    y_test = data_proc.transform.pad_truncate(test_labels, max_labels_length, position=0.0, value=0)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    print("Number of classes: ", num_classes)
    print("Number of samples: ", np.shape(train_sequences)[0] + np.shape(test_sequences)[0])

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    latent_dim = {{choice([128, 256, 512, 1024])}}
    activation = {{choice(['softmax', 'tanh', 'relu', 'sigmoid', 'linear'])}}
    dropout_rate = {{uniform(0, 1)}}
    recurrent_dropout = {{uniform(0, 1)}}
    learning_rate = {{uniform(0, 1)}}
    decay = {{uniform(0, 1)}}
    optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}

    encoder_inputs = Input(shape=(1572, len(config.channels)))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    num_classes = 31
    decoder_inputs = Input(shape=(None, num_classes))
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=recurrent_dropout)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # no attention mechanism
    decoder_dense = Dense(num_classes, activation=activation)
    decoder_pred = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_pred)

    if optimizer == 'adam':
        model.compile(optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'sgd':
        model.compile(optimizer=optimizers.SGD(lr=learning_rate, decay=decay),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'rmsprop':
        model.compile(optimizer=optimizers.RMSprop(lr=learning_rate, decay=decay),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = {{choice([20, 40, 60, 80, 100])}}
    epochs = {{choice([5, 10, 20, 50, 100, 200])}}

    print("Model config:")
    print(model.get_config())

    try:

        result = model.fit([x_train, x_test], x_test,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_split=0.1)

        # get the highest validation accuracy of the training epochs
        validation_acc = np.amax(result.history['val_acc'])
        validation_loss = np.amax(result.history['val_loss'])
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}

    except Exception as e:
        print("Error training with this model config!")
        print(e)
        K.clear_session()
        pass


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials())

    x_train, y_train, x_test, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate([y_train, y_test], y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
