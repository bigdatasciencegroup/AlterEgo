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

train_labels = tf.keras.utils.to_categorical(train_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)

print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(val_sequences)
print np.shape(val_labels)

num_classes = len(sequence_groups)


learning_rate = 1e-4
dropout_rate = 0.4

inputs = tf.placeholder(tf.float32,[None, length, np.shape(train_sequences[0])[1]]) #[batch_size,timestep,features]
targets = tf.placeholder(tf.int32, [None, num_classes])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)

conv1 = tf.layers.conv1d(inputs, 400, 3, activation=tf.nn.relu, padding='valid')
conv1 = tf.layers.max_pooling1d(conv1, 2, strides=2)
conv2 = tf.layers.conv1d(conv1, 400, 3, activation=tf.nn.relu, padding='valid')
conv2 = tf.layers.max_pooling1d(conv2, 2, strides=2)
dropout = tf.layers.dropout(conv2, dropout_rate, training=training)
reshaped = tf.reshape(dropout, [-1, np.prod(dropout.shape[1:])])
fc1 = tf.layers.dense(reshaped, 250, activation=tf.nn.relu)
logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax)

loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


num_epochs = 200
batch_size = 50

num_training_samples = len(train_sequences)
num_batches = int(num_training_samples / batch_size)
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
    dict(name='Train Acc', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Val Loss', width=8, align='center'),
    dict(name='Val Acc', width=7, align='center'),
    dict(name='', width=0, align='center'),
    dict(name='Max Val Acc', width=7, align='center'),
]

def update_table(epoch, batch, training_loss, training_accuracy, max_validation_accuracy,
                 validation_loss=None, validation_accuracy=None):
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
                 training_loss or '--', training_accuracy or '--', '',
                 validation_loss or '--', validation_accuracy or '--', '',
                 max_validation_accuracy if validation_accuracy else '--')
            
with tf.Session() as session:
    tf.global_variables_initializer().run()
    
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    start_time = time.time()
    last_time = start_time
    
    max_validation_accuracy = 0.0
    for epoch in range(num_epochs):
        training_loss = 0.0
        training_accuracy = 0.0
        permutation = np.random.permutation(num_training_samples)
        train_sequences = train_sequences[permutation]
        train_labels = train_labels[permutation]
        train_weights = train_weights[permutation]
        for batch in range(num_batches):            
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            if batch == num_batches - 1:
                indices = range(batch * batch_size, num_training_samples)
            actual_batch_size = len(indices)
            batch_sequences = train_sequences[indices]
            batch_labels = train_labels[indices]
            batch_weights = train_weights[indices]
            
            update_table(epoch, batch, training_loss / (batch_size * max(1, batch)),
                         training_accuracy / (batch_size * max(1, batch)), max_validation_accuracy)
                        
            training_feed = {inputs: batch_sequences, targets: batch_labels,
                             weights: batch_weights, training: True}
            batch_loss, _ = session.run([loss, optimizer], training_feed)
            batch_accuracy = session.run(accuracy, training_feed)
                        
            training_loss += batch_loss * actual_batch_size
            training_accuracy += batch_accuracy * actual_batch_size
            
        training_loss /= num_training_samples
        training_accuracy /= num_training_samples
        
        validation_feed = {inputs: val_sequences, targets: val_labels,
                           weights: np.ones(len(val_sequences)), training: False}
        validation_loss, validation_accuracy = session.run([loss, accuracy], validation_feed)
        max_validation_accuracy = max(validation_accuracy, max_validation_accuracy)
                
        update_table(epoch, batch, training_loss, training_accuracy,
                     max_validation_accuracy, validation_loss, validation_accuracy)
        
        reprint_header = (epoch+1) % 10 == 0 and epoch < num_epochs - 1
        table.finalize(divider=not reprint_header)
        if reprint_header:
            table.print_header()
