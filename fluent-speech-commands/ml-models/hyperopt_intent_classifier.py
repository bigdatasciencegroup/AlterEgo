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
from tensorflow.python.keras.models import Model, load_model, Sequential
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
        sequence_groups = data_proc.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate,
                                                              order=order)

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
    print("num samples:", config.num_samples)
    for data_file in input_data:
        if config.files in data_file['type']:
            train_file = data_proc.process_scrambled(data_file['labels'], [config.file_path + data_file['filename']],
                                                     channels=config.channels,
                                                     sample_rate=config.sample_rate, surrounding=config.surrounding,
                                                     exclude=set([]),
                                                     num_classes=config.num_samples)
            training_files.append(train_file)

    print("Combining input files...")
    training_sequence_groups = data_proc.combine(training_files)

    test_sequence_groups = []
    print("Training sequences:")
    print(len(training_sequence_groups), " sequences")
    lens = map(len, data_proc.get_inputs(training_sequence_groups)[0])
    print
    min(lens), np.mean(lens), max(lens)

    # Format into sequences and labels
    train_sequences, train_labels = data_proc.get_inputs(training_sequence_groups)
    train_sequences = transform_data(train_sequences)

    label_map = config.intents
    print("Label map:", len(label_map))

    # fix number of classes: should adapt according to input files
    num_classes = len(np.unique(reduce(lambda a, b: a + b, label_map)))

    label_map = map(lambda label_seq: tf.keras.utils.to_categorical(label_seq, num_classes=num_classes), label_map)

    train_labels = np.array(map(lambda i: label_map[i], train_labels))

    max_input_length = max(map(len, train_sequences))
    max_labels_length = max(map(len, train_labels))

    train_sequences = data_proc.transform.pad_truncate(train_sequences, max_input_length, position=0.0, value=-1e8)
    train_labels = data_proc.transform.pad_truncate(train_labels, max_labels_length, position=0.0, value=0)
    train_labels = np.mean(train_labels, axis=1)

    print(train_sequences.shape)
    print(train_labels.shape)

    print("Number of classes: ", num_classes)
    print("Number of samples: ", np.shape(train_sequences)[0])


    y_train = []
    y_test = []
    x_train = train_sequences
    x_test = train_labels

    return x_train, y_train, x_test, y_test


def create_model(x_train, x_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    K.clear_session()

    latent_dim = {{choice([64, 128, 256, 512])}}
    dropout_rate = {{uniform(0, 1)}}
    recurrent_dropout = {{uniform(0, 1)}}
    learning_rate = {{uniform(0, 0.5)}}
    decay = {{uniform(0, 1)}}
    batch_size = {{choice([20, 40, 60, 80])}}
    epochs = {{choice([5, 10, 20, 50, 100, 200])}}

    print('latent_dim:', latent_dim)
    print('dropout rate:', dropout_rate)
    print('recurrent dropout:', recurrent_dropout)
    print('learning rate:', learning_rate)
    print('decay:', decay)
    print('batch size:', batch_size)
    print('epochs:', epochs)

    max_input_length = 2250
    num_classes = 51

    model = Sequential()
    model.add(Bidirectional(
        LSTM(config.latent_dim, input_shape=(max_input_length, len(config.channels)), dropout=config.dropout_rate,
             recurrent_dropout=config.recurrent_dropout)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=config.learning_rate, decay=config.decay), metrics=['accuracy'])

    try:

        result = model.fit(x_train, x_test,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_split=0.1)

        # get the highest validation accuracy of the training epochs
        validation_acc = np.amax(result.history['val_acc'])
        validation_loss = np.amax(result.history['val_loss'])
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': validation_loss, 'status': STATUS_OK}

    except Exception as e:
        print("Error training with this model config!")
        print(e)
        K.clear_session()
        pass


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())

    #x_train, y_train, x_test, y_test = data()
    #print("Evaluation of best performing model:")
    #print(best_model.evaluate([y_train, y_test], y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
