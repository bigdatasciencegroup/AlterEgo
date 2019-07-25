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
            train_file = data_proc.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
                                       sample_rate=config.sample_rate, surrounding=config.surrounding, exclude=set([]),
                                       num_classes=config.num_classes)
            training_files.append(train_file)
        if 'test' in data_file['filename']:
            test_file = data_proc.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
                                       sample_rate=config.sample_rate, surrounding=config.surrounding,
                                       exclude=set([]), num_classes=config.num_classes)
            test_files.append(test_file)

print "Combining input files..."
training_sequence_groups = data_proc.combine(training_files)
test_sequence_groups = data_proc.combine(test_files)

print("Training sequences:")
print(len(training_sequence_groups), " sequences")
lens = map(len, data_proc.get_inputs(training_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

print("Validation sequences:")
print(len(test_sequence_groups), "sequences")
lens = map(len, data_proc.get_inputs(test_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

# Format into sequences and labels
train_sequences, train_labels = data_proc.get_inputs(training_sequence_groups)
test_sequences, test_labels = data_proc.get_inputs(test_sequence_groups)
print(train_sequences.shape)
print(test_sequences.shape)
print(test_labels)
print("...")


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

train_sequences = data_proc.transform.pad_truncate(train_sequences, max_input_length, position=0.0, value=-1e8)
test_sequences = data_proc.transform.pad_truncate(test_sequences, max_input_length, position=0.0, value=-1e8)
train_labels = data_proc.transform.pad_truncate(train_labels, max_labels_length, position=0.0, value=0)
test_labels = data_proc.transform.pad_truncate(test_labels, max_labels_length, position=0.0, value=0)

print("Number of classes: ", num_classes)
print("Number of samples: ", np.shape(train_sequences)[0] + np.shape(test_sequences)[0])


def split_data(fold, train_sequences, train_labels, test_sequences, test_labels):

    merged_sequences = np.concatenate((train_sequences, test_sequences), axis=0)
    merged_labels = np.concatenate((train_labels, test_labels), axis=0)

    np.random.seed(fold)
    np.random.shuffle(merged_sequences)
    np.random.shuffle(merged_labels)

    test_ind = int(round(0.9 * len(merged_sequences)))

    train_seq = merged_sequences[:test_ind]
    train_lab = merged_labels[:test_ind]

    test_seq = merged_sequences[test_ind:]
    test_lab = merged_labels[test_ind:]

    return train_seq, train_lab, test_seq, test_lab


# Result logging
timeString = time.strftime("%Y%m%d-%H%M%S", time.localtime())
log_name = "{}_e{}_b{}_phon_bidir_utkarsh_CV".format(timeString, config.num_epochs, config.batch_size)
result_file = open(log_name + ".txt", "w")
# Print header
result_file.write('# HYPERPARAMETERS:\nepochs:{}\nbatch size:{}\nlatent dim:{}\nlearning rate:{}\ndecay:{}\nattention:{}\nearly stopping:{}\nfolds:{}\n'.format(config.num_epochs, config.batch_size, config.latent_dim, config.learning_rate, config.decay, config.with_attention, config.early_stopping, config.num_folds))
result_file.write('epoch, training_loss, training_acc, max_validation_accuracy, val_loss, validation_accuracy\n')


# Cross validation
cvscores = []

for fold in list(range(config.num_folds)):

    # reset model
    K.clear_session()

    if config.num_folds > 1:
        print("Fold:", fold)
        result_file.write("Fold: " +  str(fold))
        train_sequences, train_labels, test_sequences, test_labels = split_data(fold, train_sequences, train_labels, test_sequences, test_labels)
        print("Training:", len(train_sequences))
        print("Testing:", len(test_sequences))
    # if num_folds = 1, then original train/test split is kept
    else:
        print("Training:", len(train_sequences))
        print("Testing:", len(test_sequences))

    # Model: BiLSTM encoder, LSTM decoder
    # todo: run grid search to define hyperparameters
    # todo: test dropout
    # dropout_rate = 0.4

    encoder_inputs = Input(shape=(max_input_length, len(config.channels)))
    encoder = Bidirectional(LSTM(config.latent_dim, return_state=True, return_sequences=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_classes))
    decoder_lstm = LSTM(config.latent_dim * 2, return_sequences=True, return_state=True)
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
    model.compile(optimizer=optimizers.Adam(lr=config.learning_rate, decay=config.decay), loss='categorical_crossentropy', metrics=['accuracy'])

    num_training_samples = len(train_sequences)
    num_validation_samples = len(test_sequences)
    num_training_batches = int(math.ceil(float(num_training_samples) / config.batch_size))
    num_validation_batches = int(math.ceil(float(num_validation_samples) / config.batch_size))
    start_time = None
    last_time = None

    # Table display
    progress_bar_size = 20
    max_batches = max(num_training_batches, num_validation_batches)
    layout = [
        dict(name='Ep.', width=len(str(config.num_epochs)), align='center'),
        dict(name='Batch', width=2 * len(str(max_batches)) + 1, align='center'),
        #    dict(name='', width=0, align='center'),
        dict(name='Progress/Timestamp', width=progress_bar_size + 2, align='center'),
        dict(name='ETA/Elapsed', width=7, suffix='s', align='center'),
        dict(name='', width=0, align='center'),
        dict(name='Train Loss', width=8, align='center'),
        dict(name='Train Acc', width=7, align='center'),
        dict(name='', width=0, align='center'),
        dict(name='Val Loss', width=8, align='center'),
        dict(name='Val Acc', width=7, align='center'),
        dict(name='', width=0, align='center'),
        dict(name='Max Val Acc', width=7, align='center'),
    ]

    table = DynamicConsoleTable(layout)

    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    since_training = 0


    def update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
                     validation_loss=None, validation_accuracy=None, finished=False):
        global last_time
        global since_training
        num_batches = num_training_batches
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
            progress_string = time.strftime("%I:%M:%S %p", time.localtime()) + '; ' + str(
                int(start_elapsed * 10) / 10.) + 's'
            eta_string = epoch_elapsed
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
        table.update(epoch + 1, str(batch + 1) + '/' + str(num_batches),
                     progress_string, eta_string, '',
                     training_loss or '--', training_accuracy or '--', '',
                     validation_loss or '--', validation_accuracy or '--', '',
                     max_validation_accuracy if finished else '--')


    class TrainingCallbacks(Callback):
        def __init__(self):
            super(TrainingCallbacks, self).__init__()
            self.targets = None
            self.outputs = None
            #self.not_improved = 0

            self.batch_targets = tf.Variable(0., validate_shape=False)
            self.batch_outputs = tf.Variable(0., validate_shape=False)

        def on_train_begin(self, logs={}):
            global start_time
            global last_time
            table.print_header()
            start_time = time.time()
            last_time = start_time
            self.max_validation_accuracy = 0.0

        #        print 'on_train_begin', logs

        def on_epoch_begin(self, epoch, logs={}):
            #        print 'on_train_epoch_begin', logs
            self.targets = None
            self.outputs = None
            self.training_loss = 0.0
            self.training_accuracy = 0.0
            self.epoch = epoch

        def on_batch_begin(self, batch, logs={}):
            batch_size = logs['size']
            self.batch = batch
            update_table(self.epoch, self.batch, self.training_loss / (batch_size * max(1, batch)),
                         self.training_accuracy / (batch_size * max(1, batch)), self.max_validation_accuracy)

        #        print 'on_train_batch_begin', logs

        def on_batch_end(self, batch, logs={}):
            batch_targets = K.eval(self.batch_targets)
            batch_outputs = K.eval(self.batch_outputs)
            self.targets = batch_targets if self.targets is None else np.concatenate([self.targets, batch_targets], axis=0)
            self.outputs = batch_outputs if self.outputs is None else np.concatenate([self.outputs, batch_outputs], axis=0)
            #        print
            #        print 'on_train_batch_end', logs
            #        print np.shape(self.targets)
            #        print np.shape(self.outputs)
            batch_size = logs['size']

            self.training_loss += logs['loss'] * batch_size
            self.training_accuracy += logs['acc'] * batch_size

        def on_epoch_end(self, epoch, logs={}):
            #        print 'on_train_epoch_end', logs
            #        print np.shape(self.targets)
            #        print np.shape(self.outputs)
            self.training_loss /= num_training_samples
            self.training_accuracy /= num_training_samples

            validation_accuracy = logs['val_acc']
            self.max_validation_accuracy = max(validation_accuracy, self.max_validation_accuracy)

            update_table(self.epoch, self.batch, self.training_loss, self.training_accuracy,
                         self.max_validation_accuracy, logs['val_loss'], validation_accuracy, finished=True)

            reprint_header = (self.epoch + 1) % 10 == 0 and self.epoch < config.num_epochs - 1
            table.finalize(divider=not reprint_header)
            if reprint_header:
                table.print_header()

            result_file.write(
                "{}, {}, {}, {}, {}, {}\n".format(self.epoch, self.training_loss, self.training_accuracy,
                                                      self.max_validation_accuracy, logs['val_loss'], validation_accuracy))

        def on_train_end(self, logs={}):
            pass


    #        print 'on_train_end', logs

    training_callbacks = TrainingCallbacks()

    fetches = [tf.assign(training_callbacks.batch_targets, model.targets[0], validate_shape=False),
               tf.assign(training_callbacks.batch_outputs, model.outputs[0], validate_shape=False)]
    model._function_kwargs = {'fetches': fetches}

    localtime = time.localtime()
    timeString = time.strftime("%Y%m%d-%H%M%S", localtime)

    tensorboard = TensorBoard(log_dir="logs_CV/{}_f{}".format(log_name, str(fold)), histogram_freq=1, write_graph=True, write_images=False)

    if config.early_stopping is True:
        # stop training if val accuracy has not improved since X epochs
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=10, min_delta=0.05)
        # save the best model before the stopping point
        mc = ModelCheckpoint('best_model_f{}.h5'.format(fold), monitor='val_acc', mode='max', save_best_only=True,
                             verbose=0)
        model.fit([train_sequences, train_labels[:, :-1, :]], train_labels[:, 1:, :],
                  validation_split=0.1,
                  batch_size=config.batch_size, epochs=config.num_epochs,
                  callbacks=[tensorboard, training_callbacks, es, mc], verbose=2)
        # evaluate the saved best model of this fold on the test set (is stopping early)
        best_model = load_model('best_model_f{}.h5'.format(fold))
        scores = best_model.evaluate([test_sequences, test_labels], test_labels, verbose=0)
        print("Fold %i %s on test: %.2f%%" % (fold, best_model.metrics_names[1], scores[1] * 100))
        result_file.write("Fold %i %s on test: %.2f%%\n" % (fold, best_model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    else:
        # todo: why this shape for the labels??
        #model.fit([train_sequences, train_labels[:, :-1, :]], train_labels[:, 1:, :],
        model.fit([train_sequences, train_labels], train_labels,
                  validation_split=0.1,
                  batch_size=config.batch_size, epochs=config.num_epochs,
                  callbacks=[tensorboard, training_callbacks], verbose=2)
        # evaluate on the last model
        scores = model.evaluate([test_sequences, test_labels], test_labels, verbose=2)
        print("Fold %i %s on test: %.2f%%" % (fold, model.metrics_names[1], scores[1] * 100))
        result_file.write("Fold %i %s on test: %.2f%%\n" % (fold, model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    model.summary(print_fn=lambda x: result_file.write(x + '\n'))

print("Final avg acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
result_file.write("Final avg acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
