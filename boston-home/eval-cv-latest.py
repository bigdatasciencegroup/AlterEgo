import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2' # TF INFO and WARNING messages are not printed

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False # Disables printing deprecation warnings

from sklearn.utils.class_weight import compute_class_weight
from display_utils import DynamicConsoleTable
import math
import time
import os.path
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support

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
print np.array(sequence_groups).shape # (15,10)
maxes = []
mins = []
for x in sequence_groups:
    print map(len, x), '\t', max(map(len, x))
    maxes.append(max(map(len, x)))
    mins.append(min(map(len, x)))
print max(maxes), min(mins)

time.sleep(5)
length = 900

def split_data(fold, sequence_groups):

    test_seq_groups = [None] * len(sequence_groups)
    train_seq_groups = [None] * len(sequence_groups)

    selection = range(fold*2, (fold+1)*2)
    rest = list(set(range(10)) - set(selection))

    for i in range(len(sequence_groups)): # range(15)
        test_seq_groups[i] = np.array(sequence_groups[i])[selection]
        train_seq_groups[i] = np.array(sequence_groups[i])[rest]
    train_seq_groups, val_seq_groups = data.split(train_seq_groups, 2./8)
    
    train_seqs, train_labels = data.get_inputs(data.transform.pad_truncate(train_seq_groups, length))
    val_seqs, val_labels = data.get_inputs(data.transform.pad_truncate(val_seq_groups, length))
    test_seqs, test_labels = data.get_inputs(data.transform.pad_truncate(test_seq_groups, length))

    class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
    train_weights = class_weights[list(train_labels)]
    
    train_labels = tf.keras.utils.to_categorical(train_labels)
    val_labels = tf.keras.utils.to_categorical(val_labels)

    return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels, train_weights


y_predicted_probs, acc_list = [], []
precision_macro_list, precision_micro_list = [], []
recall_macro_list, recall_micro_list = [], []
f1_macro_list, f1_micro_list = [], []
cm_trial_list = []

for trial in [0,2,8,22,26]:
    # print
    # print trial
    macro_list, micro_list = [], []
    cv_scores, cm_list = [], []
    precision_macro, precision_micro = [], []
    recall_macro, recall_micro = [], []
    f1_macro, f1_micro = [], []
    for fold in range(5):
        tf.reset_default_graph()
        train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels,\
         train_weights = split_data(fold, sequence_groups)

        num_classes = 15
        # print 'Train sequences\t', np.shape(train_sequences) # (90, 1000, 8)
        # print 'Train labels\t', np.shape(train_labels) # (90, 15)
        # print 'Val sequences\t', np.shape(val_sequences) # (30, 1000, 8)
        # print 'Val labels\t', np.shape(val_labels) # (30, 15)
        # print 'Test sequences\t', np.shape(test_sequences) # (30, 1000, 8)
        # print 'Test labels\t', np.shape(test_labels) # (30, )
        
        ####################
        #### Model (MUST BE SAME AS patient_test_serial.py, patient_test_serial_trigger.py, patient_test_serial_silence.py)
        learning_rate = 1e-4
        dropout_rate = 0.7

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
        fc1 = tf.layers.dropout(fc1, 0.5, training=training)
        logits = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax)

        loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), weights))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        correct = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        ###################
        ####### TESTING

        pred_labels = []
        test_output_list = []
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            tf.global_variables_initializer().run()
            saver.restore(session, 'best-checkpoints/t_model_{}_{}.ckpt'.format(str(trial), str(fold)))

            for sequence,label in zip(test_sequences, test_labels):
                test_feed = {inputs: [sequence], training: False}
                test_output = session.run(logits, test_feed)[0]
                pred_labels.append(np.argmax(test_output))
                same = np.argmax(test_output)!=label
                test_output_list.append(test_output)
                # print 'Predicted:', np.argmax(test_output), '\t', np.max(test_output), '\t', '*'*same
                # print 'Actual:', label 
                # print

        num_correct = np.count_nonzero(np.array(pred_labels)==test_labels)
        acc = float(num_correct)/len(test_labels)
        # print 'Correct predictions: {}/{}'.format(num_correct, len(test_labels))
        # print 'Test accuracy: {} %'.format(acc), 'on {} samples'.format(len(test_labels))

        def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
            """
            This function plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            np.set_printoptions(precision=2)

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            # Only use the labels that appear in the data
            classes = classes[unique_labels(y_true, y_pred)]
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            return cm

        classes = ['hello there good morning', 'thank you i appreciate it', 'goodbye see you later', 'it was nice meeting you', 'wish you luck and success', 'how are you doing today', 'i want to sleep now', 'can you please help me', 'i am very hungry', 'going to the bathroom', 'you are welcome', 'super tired already', 'i have been doing good', 'what is your name', 'i feel sorry for that'] 
        cm_list.append(plot_confusion_matrix(np.array(test_labels, dtype=np.int64), np.array(pred_labels), np.array(classes)))
        cv_scores.append(acc)
        y_predicted_probs.append(test_output_list)

        # Calculating metrics per fold
        macro = precision_recall_fscore_support(np.array(test_labels, dtype=np.int64), np.array(pred_labels), average='macro')
        micro = precision_recall_fscore_support(np.array(test_labels, dtype=np.int64), np.array(pred_labels), average='micro')
        
        precision_macro.append(macro[0])
        recall_macro.append(macro[1])
        f1_macro.append(macro[2])

        precision_micro.append(micro[0])
        recall_micro.append(micro[1])
        f1_micro.append(micro[2])

    precision_macro_list.append(np.mean(precision_macro))
    recall_macro_list.append(np.mean(recall_macro))
    f1_macro_list.append(np.mean(f1_macro))

    precision_micro_list.append(np.mean(precision_micro))
    recall_micro_list.append(np.mean(recall_micro))
    f1_micro_list.append(np.mean(f1_micro))

    acc_list.append(np.mean(cv_scores))
    print "Final avg acc::\t{:.2f} (+/- {:.2f})".format(np.mean(cv_scores), np.std(cv_scores))
    # print np.array(y_predicted_probs).shape

    cm_trial = np.add(cm_list[0], np.add(cm_list[1], np.add(cm_list[2], np.add(cm_list[3], cm_list[4]))))
    cm_trial_list.append(cm_trial)

final_precision_macro = np.mean(precision_macro_list)
final_recall_macro = np.mean(recall_macro_list)
final_f1_macro = np.mean(f1_macro_list)

final_precision_micro = np.mean(precision_micro_list)
final_recall_micro = np.mean(recall_micro_list)
final_f1_micro = np.mean(f1_micro_list)

final_acc, variation = np.mean(acc_list), np.std(acc_list)

print
print 'Final accuracy::\t{:.2f} (+/- {:.2f})'.format(final_acc, variation)
print
print 'Final precision::\tMacro\t{:.2f}\t\tMicro\t{:.2f}'.format(final_precision_macro, final_precision_micro)
print 'Final recall::\t\tMacro\t{:.2f}\t\tMicro\t{:.2f}'.format(final_recall_macro, final_recall_micro)
print 'Final f1::\t\tMacro\t{:.2f}\t\tMicro\t{:.2f}'.format(final_f1_macro, final_f1_micro)



def plot_cm(cm, cmap=plt.cm.Blues, title=None, normalize=False):

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           # title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    # plt.setp(ax.get_yticklabels(), fontsize=13)


    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center", fontsize=18,
            color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

max_cm = cm_trial_list[np.argmax(acc_list)]
min_cm = cm_trial_list[np.argmin(acc_list)]

plot_cm(max_cm, title='Confusion Matrix corresponding to the best (accuracy = 0.89) of 5 runs for patient P2')
plot_cm(min_cm, title='Confusion Matrix corresponding to the worst (accuracy = 0.86) of 5 runs for patient P2')
plt.show()

'''
# y_test = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]*10)
y_test = tf.keras.utils.to_categorical(test_labels)


# y_score = [0]*6 + [1]*9 + [2]*10 + [3]*17 + [4]*11 + [5]*8 + [6]*12 + [7]*10\
# 	+ [8]*10 + [9]*8 + [10]*9 + [11]*12 + [12]*8 + [13]*9 + [14]*11
# y_score = np.array(y_score)
# y_score = tf.keras.utils.to_categorical(y_score)

print y_test.shape
# # print y_test
# print y_score.shape # (150,15)
# y_score = np.array(y_score[0])
# print y_score.shape

n_classes = 15

tprs_macro, tprs_micro = [], []
aucs_macro, aucs_micro = [], []
mean_fpr_macro, mean_fpr_micro = np.linspace(0,1,100), np.linspace(0,1,100)
for fold in range(5):
    # Compute ROC curve and ROC area for each class
    y_score = np.array(y_predicted_probs[fold])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# ######################################################
    tprs_micro.append(interp(mean_fpr_micro, fpr["micro"], tpr["micro"]))
    tprs_micro[-1][0] = 0.0
    aucs_micro.append(roc_auc["micro"])

    tprs_macro.append(interp(mean_fpr_macro, fpr["macro"], tpr["macro"]))
    tprs_macro[-1][0] = 0.0
    aucs_macro.append(roc_auc["macro"])


# ######################################################

    # Plot all ROC curves
    # plt.figure(1)
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(3), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    

    

#   ######################################################
mean_tpr_macro = np.mean(tprs_macro, axis=0)
mean_tpr_macro[-1] = 1.0
mean_auc_macro = auc(mean_fpr_macro, mean_tpr_macro)
std_auc_macro = np.std(aucs_macro)

mean_tpr_micro = np.mean(tprs_micro, axis=0)
mean_tpr_micro[-1] = 1.0
mean_auc_micro = auc(mean_fpr_micro, mean_tpr_micro)
std_auc_micro = np.std(aucs_micro)

plt.plot(mean_fpr_micro, mean_tpr_micro, color='deeppink', label=r'Micro-average ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_micro, std_auc_micro), lw=2, alpha=.8)
plt.plot(mean_fpr_macro, mean_tpr_macro, color='navy', label=r'Macro-average ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_macro, std_auc_macro), lw=2, alpha=.8)

#   ######################################################

plt.plot([0, 1], [0, 1], 'k--', color=(0.6,0.6,0.6), lw=1, alpha=0.8, label='Chance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - patient P2')
plt.legend(loc="lower right")
plt.show()
'''

