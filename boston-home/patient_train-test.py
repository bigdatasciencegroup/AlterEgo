import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from display_utils import DynamicConsoleTable
import math
import time
import os.path
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import data

abs_path = os.path.abspath(os.path.dirname(__file__))

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

    def normalize_kernel(kernel, subtract_mean=False):
        if subtract_mean:
            kernel = np.array(kernel, np.float32) - np.mean(kernel)
        return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
    def ricker_function(t, sigma):
        return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
    def ricker_wavelet(n, sigma):
        return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

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
    
    return sequence_groups

        
#### Load data
def dataset(**kwargs):
    patient_dir = 'patient_data/tina'
#    patient_dir = 'patient1'
    files = map(lambda x: patient_dir + '/' + x, filter(lambda x: '.txt' in x, os.listdir(patient_dir)))
    files.sort()
    print files
    return data.join([data.process(1, [file], **kwargs) for file in files])

# channels = range(1, 8) # DO NOT CHANGE
channels = range(0, 8)

total_data = dataset(channels=channels, surrounding=200)
print np.array(total_data).shape # (Files, Samples per file)
sequence_groups = transform_data(total_data)
print len(sequence_groups) # no. of Files
print map(len, sequence_groups) # List of samples per file

# Split sequence_groups into training and validation data
training_sequence_groups, validation_sequence_groups = data.split(sequence_groups, 1./5) # test-train split per file
training_sequence_groups, test_sequence_groups = data.split(training_sequence_groups, 1./4)

print np.array(sequence_groups).shape # (15,10)
print np.array(training_sequence_groups).shape # (15,6)
print np.array(validation_sequence_groups).shape # (15,2)
print np.array(test_sequence_groups).shape # (15,2)
maxes = []
mins = []
for x in sequence_groups:
    print map(len, x)
    maxes.append(max(map(len, x)))
    mins.append(min(map(len, x)))
print max(maxes), min(mins)


# Manually selecting different training and validation datasets
#training_sequence_groups = transform_data(data.digits_session_1_dataset())
#validation_sequence_groups = transform_data(data.digits_session_4_dataset())

# Pads or truncates each sequence to length
length = 1000
training_sequence_groups = data.transform.pad_truncate(training_sequence_groups, length)
validation_sequence_groups = data.transform.pad_truncate(validation_sequence_groups, length)
test_sequence_groups = data.transform.pad_truncate(test_sequence_groups, length)

# Format into sequences and labels
train_sequences, train_labels = data.get_inputs(training_sequence_groups)
val_sequences, val_labels = data.get_inputs(validation_sequence_groups)
test_sequences, test_labels = data.get_inputs(test_sequence_groups)

print train_sequences.shape, train_labels.shape # (90, 1000, 8) (90,)
print val_sequences.shape, val_labels.shape # (30, 1000, 8) (30,)
print test_sequences.shape, test_labels.shape # (30, 1000, 8) (30,)
print
print train_labels
print val_labels
print test_labels

# Calculate sample weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
train_weights = class_weights[list(train_labels)]
# print class_weights # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# print train_weights # [1.] * 90

train_labels = tf.keras.utils.to_categorical(train_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)

print 'Train sequences', np.shape(train_sequences) # (90, 1000, 8)
print 'Train labels', np.shape(train_labels) # (90, 15)
print 'Val sequences', np.shape(val_sequences) # (30, 1000, 8)
print 'Val labels', np.shape(val_labels) # (30, 15)
print 'Test sequences', np.shape(test_sequences) # (30, 1000, 8)
print 'Test labels', np.shape(test_labels) # (30, )

num_classes = len(training_sequence_groups)
# print num_classes # 15
time.sleep(5)

####################
#### Model (MUST BE SAME AS patient_test_serial.py, patient_test_serial_trigger.py, patient_test_serial_silence.py)
learning_rate = 1e-4
dropout_rate = 0.4

inputs = tf.placeholder(tf.float32,[None, length, len(channels)]) #[batch_size,timestep,features]
targets = tf.placeholder(tf.int32, [None, num_classes])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)

conv1 = tf.layers.conv1d(inputs, 400, 12, activation=tf.nn.relu, padding='valid')
conv1 = tf.layers.max_pooling1d(conv1, 2, strides=2)
conv2 = tf.layers.conv1d(conv1, 400, 6, activation=tf.nn.relu, padding='valid')
conv2 = tf.layers.max_pooling1d(conv2, 2, strides=2)
conv3 = tf.layers.conv1d(conv2, 400, 3, activation=tf.nn.relu, padding='valid')
conv3 = tf.layers.max_pooling1d(conv3, 2, strides=2)
conv4 = tf.layers.conv1d(conv3, 400, 3, activation=tf.nn.relu, padding='valid')
conv4 = tf.layers.max_pooling1d(conv4, 2, strides=2)
conv5 = tf.layers.conv1d(conv4, 400, 3, activation=tf.nn.relu, padding='valid')
conv5 = tf.layers.max_pooling1d(conv5, 2, strides=2)
dropout = tf.layers.dropout(conv5, dropout_rate, training=training)
reshaped = tf.reshape(dropout, [-1, np.prod(dropout.shape[1:])])
fc1 = tf.layers.dense(reshaped, 250, activation=tf.nn.relu)
logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax)

loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
####################

num_epochs = 700
batch_size = 50

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
    dict(name='Train Acc', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Val Loss', width=8, align='center'),
    dict(name='Val Acc', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Max Val Acc', width=7, align='center'),
]
since_training = 0
def update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
                 validation_loss=None, validation_accuracy=None, finished=False):
    global last_time
    global since_training
    num_batches = num_training_batches if validation_loss is None else num_validation_batches
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
    table.update(epoch + 1, str(batch + 1) + '/' + str(num_batches),
                 progress_string, eta_string, '',
                 training_loss or '--', training_accuracy or '--', '',
                 validation_loss or '--', validation_accuracy or '--', '',
                 max_validation_accuracy if finished else '--')



show_confusion_matrix = True
        
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    tf.global_variables_initializer().run()
    
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    start_time = time.time()
    last_time = start_time
    
    # Training/validation loop
    max_validation_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training
        training_loss = 0.0
        training_accuracy = 0.0
        permutation = np.random.permutation(num_training_samples)
        train_sequences = train_sequences[permutation]
        train_labels = train_labels[permutation]
        train_weights = train_weights[permutation]
        train_output = None
        for batch in range(num_training_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_training_batches - 1:
                indices = range(batch * batch_size, num_training_samples)
            batch_sequences = train_sequences[indices]
            batch_labels = train_labels[indices]
            batch_weights = train_weights[indices]
            
            update_table(epoch, batch, training_loss / (batch_size * max(1, batch)),
                         training_accuracy / (batch_size * max(1, batch)), max_validation_accuracy)
                        
            training_feed = {inputs: batch_sequences, targets: batch_labels,
                             weights: batch_weights, training: True}
            batch_loss, _, batch_output = session.run([loss, optimizer, logits], training_feed)
            batch_accuracy = session.run(accuracy, training_feed)
                        
            training_loss += batch_loss * len(indices)
            training_accuracy += batch_accuracy * len(indices)
            train_output = batch_output if train_output is None else \
                                    np.concatenate([train_output, batch_output], axis=0)
            
        training_loss /= num_training_samples
        training_accuracy /= num_training_samples
        
        # Validation
        validation_loss = 0.0
        validation_accuracy = 0.0
        val_output = None
        for batch in range(num_validation_batches):         
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_validation_batches - 1:
                indices = range(batch * batch_size, num_validation_samples)
            batch_sequences = val_sequences[indices]
            batch_labels = val_labels[indices]
            batch_weights = np.ones(len(batch_sequences))
    
            update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
                         validation_loss / (batch_size * max(1, batch)),
                         validation_accuracy / (batch_size * max(1, batch)))
            
            validation_feed = {inputs: batch_sequences, targets: batch_labels,
                               weights: batch_weights, training: False}
            batch_loss, batch_accuracy, batch_output = session.run([loss, accuracy, logits], validation_feed)
            validation_loss += batch_loss * len(indices)
            validation_accuracy += batch_accuracy * len(indices)
            val_output = batch_output if val_output is None else np.concatenate([val_output, batch_output], axis=0)
            
        validation_loss /= num_validation_samples
        validation_accuracy /= num_validation_samples
        if validation_accuracy > max_validation_accuracy:
            model_name = 'checkpoints/t_model.ckpt'
            save_path = saver.save(session, os.path.join(abs_path, model_name))
            best_epoch = epoch
            print ' Model saved:', model_name,
        max_validation_accuracy = max(validation_accuracy, max_validation_accuracy)
        
        update_table(epoch, batch, training_loss, training_accuracy,
                     max_validation_accuracy, validation_loss, validation_accuracy, finished=True)
        
        # Also ignore this part
        if show_confusion_matrix:
            predicted = np.argmax(val_output, axis=1)
            actual = np.argmax(val_labels, axis=1)
            val_output = [[] for _ in range(num_classes)]
            val_count_matrix = [[0] * num_classes for _ in range(num_classes)]
            for i in range(len(actual)):
                val_output[actual[i]].append(predicted[i])
                val_count_matrix[actual[i]][predicted[i]] += 1
            predicted = np.argmax(train_output, axis=1)
            actual = np.argmax(train_labels, axis=1)
            train_output = [[] for _ in range(num_classes)]
            train_count_matrix = [[0] * num_classes for _ in range(num_classes)]
            for i in range(len(actual)):
                train_output[actual[i]].append(predicted[i])
                train_count_matrix[actual[i]][predicted[i]] += 1
            print
            print
            train_max_length = max(2, len(str(max(map(len, training_sequence_groups)))))
            val_max_length = max(2, len(str(max(map(len, validation_sequence_groups)))))
            print 'TRAINING', ' ' * ((num_classes + 1) * (train_max_length + 1) + 1), '\t\t\t', 'VALIDATION'
            print ' ' * (train_max_length - 2) + 'Predicted ', ''.join(map(
                    lambda x: str(x) + ' ' * (train_max_length - len(str(x)) + 1), range(num_classes))), '\t\t\t', \
                    ' ' * (val_max_length - 2) + 'Predicted ', ''.join(map(
                    lambda x: str(x) + ' ' * (val_max_length - len(str(x)) + 1), range(num_classes)))
            print 'Actual\t', '-' * ((num_classes + 1) * (train_max_length + 1) + 1), '\t\t\t', \
                    'Actual\t', '-' * ((num_classes + 1) * (val_max_length + 1) + 1)
            for i in range(num_classes):
                print ' '*(5-len(str(i))), i, '\t|' + ' ' * (train_max_length - 1), \
                    ''.join(map(lambda x: (str(x) if x else '-') + ' ' * (train_max_length - len(str(x)) + 1), \
                                train_count_matrix[i])) + '| ', \
                    str.format('{0:.5f}', np.mean(np.array(train_output[i]) == i)), \
                    '\t\t', ' '*(5-len(str(i))), i, '\t|' + ' ' * (val_max_length - 1), \
                    ''.join(map(lambda x: (str(x) if x else '-') + ' ' * (val_max_length - len(str(x)) + 1), \
                                val_count_matrix[i])) + '| ', \
                    str.format('{0:.5f}', np.mean(np.array(val_output[i]) == i))
            print '\t', '-' * ((num_classes + 1) * (train_max_length + 1) + 1), '\t\t\t', \
                    '\t', '-' * ((num_classes + 1) * (val_max_length + 1) + 1)
        
        reprint_header = ((epoch+1) % 10 == 0 or show_confusion_matrix) and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()

print 'Best checkpoint saved in epoch # {}'.format(best_epoch+1)

####### TESTING

pred_labels = []
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    tf.global_variables_initializer().run()
    saver.restore(session, 'checkpoints/t_model.ckpt')

    for sequence,label in zip(test_sequences, test_labels):
        test_feed = {inputs: [sequence], training: False}
        test_output = session.run(logits, test_feed)[0]
        pred_labels.append(np.argmax(test_output))
        same = np.argmax(test_output)!=label
        print 'Predicted:', np.argmax(test_output), '\t', np.max(test_output), '\t', '*'*same
        print 'Actual:', label 
        print

num_correct = np.count_nonzero(np.array(pred_labels)==test_labels)
acc = float(num_correct)/len(test_labels)*100
print 'Correct predictions: {}/{}'.format(num_correct, len(test_labels))
print 'Test accuracy: {} %'.format(acc), 'on {} samples'.format(len(test_labels))

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Plotting Normalized confusion matrix")
    else:
        print("Plotting Confusion matrix, without normalization")

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

classes = ['hello there good morning', 'thank you i appreciate it', 'goodbye see you later', 'it was nice meeting you', 'wish you luck and success', 'how are you doing today', 'i want to sleep now', 'can you please help me', 'i am very hungry', 'going to the bathroom', 'you are welcome', 'super tired already', 'i have been doing good', 'what is your name', 'i feel sorry for that'] 
plot_confusion_matrix(np.array(test_labels, dtype=np.int64), np.array(pred_labels), np.array(classes))
plt.show()
