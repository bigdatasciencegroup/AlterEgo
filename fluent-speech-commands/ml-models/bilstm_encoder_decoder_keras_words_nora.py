# Set seed value
seed_value = 123
import numpy as np
# Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
import tensorflow as tf
# Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)
import os
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, CuDNNLSTM, LSTM, Dense, Concatenate, Bidirectional
from tensorflow.python.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
import matplotlib
import time
import data_proc
import config
import json
from attention_keras.layers.attention import AttentionLayer

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
    sequence_groups = data_proc.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate,
                                                          order=order)

    #### Apply hard bandpassing
    # sequence_groups = data.transform.fft(sequence_groups)
    # sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    # sequence_groups = np.real(data.transform.ifft(sequence_groups))
    #
    return sequence_groups


# Result logging
print("SESSION:")
print(config.files)
timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
log_name = "{}_e{}_b{}_sents_bidir_nora_CV_{}".format(timeString, config.num_epochs, config.batch_size, config.files[-1])
result_file = open(log_name + ".txt", "w")
result_file.write(str(config.files))

# Read config file
with open(config.data_maps, 'r') as f:
    input_data = json.load(f)

# Read input files
training_files = []
for data_file in input_data:
    if any(sess in data_file['type'] for sess in config.files):
    #if 'sentences_nora_sess1' in data_file['type'] or 'sentences_nora_sess2' in data_file['type'] or 'sentences_nora_sess3' in data_file['type'] or 'sentences_nora_sess4' in data_file['type'] or 'sentences_nora_sess5' in data_file['type'] or 'sentences_nora_sess6' in data_file['type'] or 'sentences_nora_sess7' in data_file['type'] or 'sentences_utkarsh_sess8' in data_file['type']:
        train_file = data_proc.process_scrambled(data_file['labels'], [config.file_path + data_file['filename']],
                                                 channels=config.channels,
                                                 sample_rate=config.sample_rate, surrounding=config.surrounding,
                                                 exclude=set([]),
                                                 num_classes=config.num_samples)
        training_files.append(train_file)

print "Combining input files..."
training_sequence_groups = data_proc.combine(training_files)

print("Training sequences:")
print(len(training_sequence_groups), " sequences")
lens = map(len, data_proc.get_inputs(training_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

# Format into sequences and labels
train_sequences, train_labels = data_proc.get_inputs(training_sequence_groups)
print(train_sequences.shape)

train_sequences = transform_data(train_sequences)
print(len(train_sequences))

result_file.write('\nsentences_label_map_caseInsensitive_splitContract\n')
label_map = config.sentences_label_map_caseInsensitive_splitContract
label_map = label_map[:len(train_sequences)]
print("Label map:", len(label_map))

# fix number of classes: should adapt according to input files
num_classes = len(np.unique(reduce(lambda a, b: a + b, label_map))) + 2  # (for start and end symbols)
start_symbol = num_classes - 2
end_symbol = num_classes - 1

label_map = map(lambda label_seq: [start_symbol] + label_seq + [end_symbol], label_map)
label_map = map(lambda label_seq: tf.keras.utils.to_categorical(label_seq, num_classes=num_classes), label_map)

train_labels = np.array(map(lambda i: label_map[i], train_labels))

max_input_length = max(map(len, train_sequences))
max_labels_length = max(map(len, train_labels))
result_file.write('\nMax input length:{}\n'.format(max_input_length))

train_sequences_all = data_proc.transform.pad_truncate(train_sequences, max_input_length, position=0.0, value=-1e8)
train_labels_all = data_proc.transform.pad_truncate(train_labels, max_labels_length, position=0.0, value=0)

print("Number of classes: ", num_classes)
print("Number of samples: ", np.shape(train_sequences)[0])


def split_data(num_fold, all_sequences, all_labels):

    # shuffle sequences and labels equally
    indices = np.arange(all_sequences.shape[0])
    print(indices)
    np.random.seed(num_fold)
    np.random.shuffle(indices)
    print(indices)
    all_sequences = all_sequences[indices]
    all_labels = all_labels[indices]

    test_ind = int(round(0.9 * len(all_sequences)))

    train_seq = all_sequences[:test_ind]
    train_lab = all_labels[:test_ind]

    test_seq = all_sequences[test_ind:]
    test_lab = all_labels[test_ind:]

    return train_seq, train_lab, test_seq, test_lab


# Print parameters to log file
result_file.write(
    '# HYPERPARAMETERS:\nsurrounding:{}\nepochs:{}\nbatch size:{}\nlatent dim:{}\nlearning rate:{}\ndecay:{}\ndropout rate:{}\nrecurrent dropout:{}\nattention:{}\nearly stopping:{}\nfolds:{}\nsentences:{}\n'.format(
        config.surrounding, config.num_epochs, config.batch_size, config.latent_dim, config.learning_rate, config.decay,
        config.dropout_rate, config.recurrent_dropout, config.with_attention, config.early_stopping, config.num_folds,
        config.num_samples))

# Cross validation
cv_scores = []

for fold in list(range(config.num_folds)):

    # reset model
    K.clear_session()
    # Configure a new global `tensorflow` session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # split data into train and test
    print("Fold:", fold)
    result_file.write("Fold: " + str(fold))
    train_seqs, train_labs, test_seqs, test_labs = split_data(fold, train_sequences_all, train_labels_all)
    print("Training:", len(train_seqs))
    print("Testing:", len(test_seqs))

    encoder_inputs = Input(shape=(max_input_length, len(config.channels)))
    encoder = Bidirectional(LSTM(config.latent_dim, return_state=True, return_sequences=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_classes))
    decoder_lstm = LSTM(config.latent_dim * 2, return_sequences=True, return_state=True, dropout=config.dropout_rate,
                        recurrent_dropout=config.recurrent_dropout)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    if config.with_attention is True:
        # Using Bahdanau attention for MT: https://arxiv.org/pdf/1409.0473.pdf
        # Code from: https://github.com/thushv89/attention_keras
        attn_layer = AttentionLayer(name='attention_layer')
        attn_outputs, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_outputs])
        decoder_dense = Dense(num_classes, activation=config.activation_function)
        decoder_pred = decoder_dense(decoder_concat_input)
    else:
        # no attention mechanism
        decoder_dense = Dense(num_classes, activation=config.activation_function)
        decoder_pred = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_pred)
    model.compile(optimizer=optimizers.SGD(lr=config.learning_rate, decay=config.decay),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizers.Adam(lr=config.learning_rate, decay=config.decay),
     #             loss='categorical_crossentropy', metrics=['accuracy'])

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(config.latent_dim * 2,))
    decoder_state_input_c = Input(shape=(config.latent_dim * 2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    tensorboard = TensorBoard(log_dir="logs_CV/{}_f{}".format(log_name, str(fold)), histogram_freq=1, write_graph=True,
                              write_images=False)

    if config.early_stopping is True:
        # stop training if val accuracy has not improved since X epochs
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=8, min_delta=0.005)
        # save the best model before the stopping point
        mc = ModelCheckpoint('best_model_{}_f{}.h5'.format(log_name, str(fold)), monitor='val_loss', mode='min', save_best_only=True,
                             verbose=0)
        history = model.fit([train_seqs, train_labs[:, :-1, :]], train_labs[:, 1:, :],
                            validation_split=0.1, shuffle=False,
                            batch_size=config.batch_size, epochs=config.num_epochs,
                            callbacks=[tensorboard, es, mc], verbose=1)

        result_file.write("\nFinal train acc:%.2f%%, train loss:%.4f%%, val acc:%.2f%%, val loss:%.4f%%\n" % (
            history.history['acc'][-1], history.history['loss'][-1], history.history['val_acc'][-1],
            history.history['val_loss'][-1]))

        scores = model.evaluate([test_seqs, test_labs[:, :-1, :]], test_labs[:, 1:, :], verbose=1)
        print("Fold %i %s on test: %.2f%%" % (fold, model.metrics_names[1], scores[1] * 100))
        result_file.write("\nFold %i %s on test: %.2f%%\n" % (fold, model.metrics_names[1], scores[1] * 100))

        # evaluate the saved best model of this fold on the test set (is stopping early)
        best_model = load_model('best_model_{}_f{}.h5'.format(log_name, str(fold)))
        scores = best_model.evaluate([test_seqs, test_labs[:, :-1, :]], test_labs[:, 1:, :], verbose=2)
        print("Fold %i %s on test: %.2f%%" % (fold, best_model.metrics_names[1], scores[1] * 100))
        result_file.write("Fold %i %s on test: %.2f%%\n" % (fold, best_model.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)

    else:
        print(train_seqs.shape)
        print(train_labs.shape)
        history = model.fit([train_seqs, train_labs[:, :-1, :]], train_labs[:, 1:, :],
                            validation_split=0.1, shuffle=False,
                            batch_size=config.batch_size, epochs=config.num_epochs,
                            callbacks=[tensorboard], verbose=1)
        result_file.write("\nTrain acc:%.2f%%, train loss:%.4f%%, val acc:%.2f%%, val loss:%.4f%%\n" % (
            history.history['acc'][-1], history.history['loss'][-1], history.history['val_acc'][-1],
            history.history['val_loss'][-1]))

        # evaluate on the last model
        scores = model.evaluate([test_seqs, test_labs[:, :-1, :]], test_labs[:, 1:, :], verbose=1)
        print("Fold %i %s on test: %.2f%%" % (fold, model.metrics_names[1], scores[1] * 100))
        result_file.write("\nFold %i %s on test: %.2f%%\n" % (fold, model.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)

    # save encoder, decoder and full model
    model.save('SavedModels/Full_{}.h5'.format(log_name))
    print('Model Full_{}.h5 saved in SavedModels'.format(log_name))
    encoder_model.save('SavedModels/Encoder_{}.h5'.format(log_name))
    print('Model Encoder_{}.h5 saved in SavedModels'.format(log_name))
    decoder_model.save('SavedModels/Decoder_{}.h5'.format(log_name))
    print('Model Decoder_{}.h5 saved in SavedModels'.format(log_name))

print("Final avg test acc: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
result_file.write("Final avg test acc: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
