import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# from nltk.translate.bleu_score import sentence_bleu

import math
import statistics
import time
import json

# Local imports
import config
import data_proc

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
    # sequence_groups = data_proc.transform.normalize_std(sequence_groups)

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
    # sequence_groups = data_proc.transform.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5  # 0.5
    high_freq = 8  # 8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = data_proc.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

    #### Apply hard bandpassing
    # sequence_groups = data_proc.transform.fft(sequence_groups)
    # sequence_groups = data_proc.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    # sequence_groups = np.real(data_proc.transform.ifft(sequence_groups))
    #
    return sequence_groups


with open(config.data_maps, 'r') as f:
    input_data = json.load(f)

training_files = []
test_files = []
# for data_file in input_data:
#     if data_file['type'] == 'phonemes_utkarsh':
#         if 'train' in data_file['filename']:
#             train_file = data_proc.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
#                                        sample_rate=config.sample_rate, surrounding=config.surrounding, exclude=set([]),
#                                        num_classes=config.num_classes)
#             training_files.append(train_file)
#         if 'test' in data_file['filename']:
#             test_file = data_proc.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
#                                        sample_rate=config.sample_rate, surrounding=config.surrounding,
#                                        exclude=set([]), num_classes=config.num_classes)
#             training_files.append(test_file)

for data_file in input_data:
    if data_file['type'] == 'phonemes_common_utkarsh_s1':
        train_file = data_proc.process_scrambled(data_file['labels'], [config.file_path+data_file['filename']], channels=config.channels,
                                   sample_rate=config.sample_rate, surrounding=config.surrounding, exclude=set([]),
                                   num_classes=config.num_classes)
        training_files.append(train_file)

training_sequence_groups = data_proc.combine(training_files)

print("Training sequences:")
print(len(training_sequence_groups), " sequences")
lens = map(len, data_proc.get_inputs(training_sequence_groups)[0])
print min(lens), np.mean(lens), max(lens)

# Format into sequences and labels
train_sequences, train_labels = data_proc.get_inputs(training_sequence_groups)
train_sequences = transform_data(train_sequences)

label_map = config.phoneme_label_map
print("Label map:", len(label_map))
num_classes = len(np.unique(reduce(lambda a,b: a+b, label_map))) + 2 #(for start and end symbols)
start_symbol = num_classes - 2 # 29
end_symbol = num_classes - 1 # 30

label_map = map(lambda label_seq: [start_symbol] + label_seq + [end_symbol], label_map)
label_map = map(lambda label_seq: tf.keras.utils.to_categorical(label_seq, num_classes=num_classes), label_map)

train_labels = np.array(map(lambda i: label_map[i], train_labels))

max_input_length = max(map(len, train_sequences))
max_labels_length = max(map(len, train_labels))

train_sequences_all = data_proc.transform.pad_truncate(train_sequences, max_input_length, position=0.0, value=-1e8)
train_labels_all = data_proc.transform.pad_truncate(train_labels, max_labels_length, position=0.0, value=0)

def split_data(num_fold, all_sequences, all_labels):

    indices = np.arange(all_sequences.shape[0])
    # print(indices)
    np.random.seed(num_fold)
    np.random.shuffle(indices)
    # print(indices)

    all_sequences = all_sequences[indices]
    all_labels = all_labels[indices]

    test_ind = int(round(0.9 * len(all_sequences)))

    train_seq = all_sequences[:test_ind]
    train_lab = all_labels[:test_ind]

    test_seq = all_sequences[test_ind:]
    test_lab = all_labels[test_ind:]

    return train_seq, train_lab, test_seq, test_lab

train_sequences, train_labels, test_sequences, test_labels = split_data(0, train_sequences_all, train_labels_all)

# --------------------------------------------------------------------------------------------------------------------------------------------

def batch_greedy_decode(input_seq, encoder_model, decoder_model, max_decoder_seq_length, start_symbol, end_symbol, num_classes):
    '''
    Performs batch greedy decode on the input_seq batch until max_decoder_seq_length while dynamically
    pruning samples that've reached end_symbol already.
    Returns final_predicted_labels (list of arrays) correspondong to input_seq
    '''
    print '\nDecoding.............................'
    if input_seq.ndim==2:   input_seq = np.expand_dims(input_seq, 0)
    samples = input_seq.shape[0]

    states_value = encoder_model.predict(input_seq) # Encoder states

    # Generate empty target sequence of length 1    
    target_seq = np.zeros((samples, 1, num_classes))
    target_seq[:, 0, start_symbol] = 1

    decoded_seq_list = np.asarray([[start_symbol]]*samples) # completed seqs are dynamically pruned
    output_seq_list = [] # completed seqs are dynamically appended
    final_locations = np.empty(shape=(0,)) # will track corresponding locations for decoded sequences
    sample_locations = np.arange(samples)

    i = 1

    while i<max_decoder_seq_length:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)

        decoded_seq_list = np.append(decoded_seq_list, np.expand_dims(sampled_token_index,1), axis=-1)

        if np.any(sampled_token_index[:]==end_symbol):
            locations = np.where(sampled_token_index[:]==end_symbol)

            # Appending
            output_seq_list.append(decoded_seq_list[locations[0]])
            final_locations = np.append(final_locations, sample_locations[locations[0]])
            # final_locations = np.append(final_locations, locations[0])

            # Pruning
            decoded_seq_list = np.delete(decoded_seq_list, locations[0], axis=0) # remove completed sequences
            sample_locations = np.delete(sample_locations, locations[0])
            sampled_token_index = np.delete(sampled_token_index, locations[0]) # to update target_seq
            h = np.delete(h, locations[0], axis=0) # fed into decoder
            c = np.delete(c, locations[0], axis=0) # fed into decoder
            
            samples-=locations[0].shape[0]
            if samples==0:
                break

        # Update the target sequence, fed into decoder
        target_seq = np.zeros((samples, 1, num_classes))
        target_seq[np.arange(samples), 0, sampled_token_index] = 1
        
        # Update states
        states_value = [h, c]
        i+=1

    # All remaining (that didn't output end_symbol but reached max length)
    output_seq_list.append(decoded_seq_list[:])
    final_locations = np.append(final_locations, sample_locations[:])
    
    # print('FINAL LOCS', final_locations.shape)
    # print(final_locations)

    # output_seq_list is a list of 2d arrays with shapes (x,y) where both x and y are variables
    # hence can't be concatenated together

    predicted_labels = []
    for i in output_seq_list:
        for j in i:
            if len(j)!=0:   predicted_labels.append(j) # to disregard empty array

    # print('PREDICTED LABELS', len(predicted_labels))
    # print(predicted_labels)

    # predicted_labels is a list of 1d arrays with shape (x,) where x is variable
    # But the final_locations aren't arranged yet

    # Arranging predicted_labels into final_predicted_labels while sorting through final_locations
    final_predicted_labels = []
    for i in range(input_seq.shape[0]):
        index = np.where(final_locations==i)[0]
        final_predicted_labels.append(predicted_labels[index[0]])
    
    # print '\n', final_predicted_labels
    print 'Decoding finished....................\n'

    return final_predicted_labels


def greedy_decode(input_seq, encoder_model, decoder_model, max_decoder_seq_length, start_symbol, end_symbol, num_classes):
    input_seq = np.expand_dims(input_seq, 0)
    states_value = encoder_model.predict(input_seq) # Encoder states

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_classes))
    # Populate the first character of target sequence with the start character
    target_seq[0, 0, start_symbol] = 1 # target_seq[0,0,29] = 1

    # Sampling loop for a batch of sequences
    # To simplify, we assume a batch of size 1
    stop_condition = False
    decoded_sequence = [start_symbol] # [29]
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value) # Takes in target_seq and Encoder states
        # print('OUTPUT TOKENS:', output_tokens)
        # print('SUM:', np.sum(output_tokens))

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) # Greedy search decoding
        decoded_sequence.append(sampled_token_index) # [29,loc1]

        # Exit condition: either hit max length
        # or find stop character
        if (sampled_token_index == end_symbol or len(decoded_sequence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, num_classes))
        target_seq[0, 0, sampled_token_index] = 1

        # Update states
        states_value = [h, c]

    return decoded_sequence


def beam_decode(input_seq, encoder_model, decoder_model, max_decoder_seq_length, start_symbol, end_symbol, num_classes, k=5):

    output_seq_list = []
    output_probs_list = []

    input_seq = np.expand_dims(input_seq, 0)
    h, c = encoder_model.predict(input_seq) # Encoder states

    decoded_seq_list = np.zeros((k, max_decoder_seq_length), dtype=np.int32)
    decoded_seq_list[:, 0] = start_symbol
    decoded_probs = np.ones(k, dtype=np.float32)

    target_seq_list = np.zeros((k, 1, num_classes)) # Generate empty target sequences
    target_seq_list[:, 0, start_symbol] = 1 # Populate the first character of target sequence with the start character

    h_values_list = np.tile(h, k).reshape((k, h.shape[1]))
    c_values_list = np.tile(c, k).reshape((k, c.shape[1]))

    i = 1 # i!=0 coz start symbol already given

    while k>0 and i<max_decoder_seq_length:
        print('-'*130)
        output_tokens, h, c = decoder_model.predict([target_seq_list, h_values_list, c_values_list]) # Takes in target_seq and Encoder states
        output_tokens = np.squeeze(output_tokens, axis=1)
        # print('OUTPUT TOKENS:', output_tokens.shape)
        # print('OUTPUT TOKENS:', output_tokens)

        # Calc  probs
        top_k_output_tokens_ind = np.argsort(output_tokens, axis=1)[:,-k:]
        top_k_output_tokens_prob =  np.sort(output_tokens, axis=1)[:,-k:]
        decoded_probs_diagonal = np.diag(decoded_probs)
        print decoded_probs_diagonal.shape, top_k_output_tokens_prob.shape
        k_probs = np.dot(decoded_probs_diagonal, top_k_output_tokens_prob)

        # Update the target sequence (of length 1)
        flatten_k_probs = np.argsort(k_probs.flatten())[-k:]
        rows = flatten_k_probs/k # branches
        cols = flatten_k_probs%k

        # First character
        if i==1:
            rows = np.arange(k)
            cols = np.arange(k)

        selected_token_ind = top_k_output_tokens_ind[rows,cols]

        print(rows)
        print(selected_token_ind)

        target_seq_list = np.zeros((k,1,num_classes))
        target_seq_list[np.arange(k),0,selected_token_ind]=1

        decoded_temp_list = decoded_seq_list[rows]
        decoded_temp_list[:,i] = selected_token_ind
        decoded_seq_list = decoded_temp_list
        decoded_probs = k_probs.flatten()[flatten_k_probs]

        # Update states
        h_values_list = h[rows,:]
        c_values_list = c[rows, :]
        i+=1

        # Delete finished branches
        to_be_deleted = []
        # for a,b in zip(rows,cols):
        #     if top_k_output_tokens_ind[a,b]==end_symbol:
        #         output_seq_list.append(list(decoded_seq_list[a]))
        #         output_probs_list.append(decoded_probs[a])
        #         to_be_deleted.append(a)
        idx_list = []
        for idx,(a,b) in enumerate(zip(rows,cols)):
            if top_k_output_tokens_ind[a,b]==end_symbol:
                output_seq_list.append(list(decoded_seq_list[a]))
                output_probs_list.append(decoded_probs[a])
                to_be_deleted.append(a)
                idx_list.append(idx)
        target_seq_list = np.delete(target_seq_list, idx_list, axis=0)
        h_values_list = np.delete(h_values_list, idx_list, axis=0)
        c_values_list = np.delete(c_values_list, idx_list, axis=0)

        '''
        to_be_deleted = rows[np.where(top_k_output_tokens_ind[rows,cols]==end_symbol)]
        output_seq_list += decoded_seq_list[to_be_deleted]
        output_probs_list += decode_probs[to_be_deleted]'''

        decoded_seq_list = np.delete(decoded_seq_list, to_be_deleted, axis=0)
        decoded_probs = np.delete(decoded_probs, to_be_deleted, axis=0)
        k-=len(to_be_deleted)

    # append the rest
    if k>0:
        output_seq_list.append(list(decoded_seq_list))
        output_probs_list.append(list(decoded_probs))

    return output_seq_list, output_probs_list


def plot_confusion_matrix(sequences, labels, classes, encoder_model, decoder_model, max_decoder_seq_length,\
 start_symbol, end_symbol, num_classes, decode='batch_greedy_decode', normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    actual_labels = []
    for label in np.argmax(labels, axis=-1).tolist():
        for idx, val in enumerate(label):
            if val==end_symbol:     actual_labels.append(label[:idx+1])
    if decode=='batch_greedy_decode':
        predicted_labels = [list(x) for x in batch_greedy_decode(sequences, encoder_model, decoder_model,\
         max_decoder_seq_length, start_symbol, end_symbol, num_classes)]

    counter = 0 # Tracks number of predictions with same length as actual target
    y_true, y_pred = [], []

    for index in range(len(sequences)):
        if (len(actual_labels[index]) == len(predicted_labels[index])):
            for x in list(actual_labels[index]):
                if x!=start_symbol and x!=end_symbol:       y_true.append(x)
            for x in list(predicted_labels[index]):
                if x!=start_symbol and x!=end_symbol:       y_pred.append(x)
            counter+=1
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print('Sequences considered for confusion matrix : {} out of {}'.format(counter, len(sequences)))
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


def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_predictions(true_labels_file, predicted_labels_file, sequences, labels, classes, encoder_model, decoder_model,\
 max_decoder_seq_length, start_symbol, end_symbol, num_classes, decode='batch_greedy_decode'):

    actual_labels = []
    for label in np.argmax(labels, axis=-1).tolist():
        for idx, val in enumerate(label):
            if val==end_symbol:     actual_labels.append(label[:idx+1])

    if decode == 'batch_greedy_decode':
        predicted_labels = batch_greedy_decode(sequences, encoder_model, decoder_model, \
                                               max_decoder_seq_length, start_symbol, end_symbol, num_classes)

    print(len(actual_labels))
    print(len(predicted_labels))

    for sample_true, sample_pred in zip(actual_labels, predicted_labels):

        #print(sample_true) 
        for x in sample_true[1:]:
            if x != end_symbol:
                true_labels_file.write(classes[x] + " ")
            else:
                true_labels_file.write("\n")
                break

        #print(sample_pred)
        for x in sample_pred[1:]:
            if x != end_symbol:
                predicted_labels_file.write(classes[x] + " ")
            else:
                predicted_labels_file.write("\n")


def error_rate(sequences, labels, encoder_model, decoder_model, max_decoder_seq_length,\
 start_symbol, end_symbol, num_classes, decode='batch_greedy_decode'):

    act_labels = []
    for label in np.argmax(labels, axis=-1).tolist():
        for idx, val in enumerate(label):
            if val==end_symbol:     act_labels.append(label[:idx+1])

    if decode=='batch_greedy_decode':
        pred_labels = batch_greedy_decode(sequences, encoder_model, decoder_model,\
         max_decoder_seq_length, start_symbol, end_symbol, num_classes)

    # print act_labels
    # print
    # print pred_labels
    # print

    norm_distance_list = []
    for index in range(len(act_labels)):
        distance = edit_distance(list(act_labels[index])[1:-1], list(pred_labels[index])[1:-1])
        norm_distance = distance/float(len(list(act_labels[index])[1:-1]))
        norm_distance_list.append(norm_distance)

    return statistics.mean(norm_distance_list)


def bit_rate(application_speed, error_rate_, vocabulary_size):
    '''bit-rate (also referred to as Wolpaw rate), as explained by Kronegg et al. and originally proposed by Wolpaw et al.
    measures the performance of human-computer interfaces.

    Returns information transfer rate in bits per minute given the 
    application speed V (average words per minute / phonemes per minute)
    classification accuracy P (1 minus word error rate / phoneme error rate)
    vocabulary size N (no. of distinct words / phonemes)
    B = V * [logN + PlogP + (1-P)log(1-P/N-1)] words per minute / phonemes per minute
    PS: All logs have base 2
    '''
    classification_accuracy = 1 - error_rate_
    return application_speed * (math.log(vocabulary_size,2) + (classification_accuracy*math.log(classification_accuracy,2)) + (error_rate_*math.log(error_rate_/(vocabulary_size-1),2)))

# --------------------------------------------------------------------------------------------------------------------------------------------


# Enter the model name, sequences, labels and class_names
log_name = '20190806-153845_e2500_b80_phon_common_utkarsh'
# log_name = '20190804-205820_e500_b80_phon_common_utkarsh' # training loss = 0.98, val loss = 1.56, per = 58.1% on training data, 97.8% on val data, 96.6% on test data
# log_name = '20190803-200253_e2500_b80_phon_common_utkarsh' # training loss = 0.24, val loss = 2.4, per =  4.3% on training data, 107.4% on val data, 99.2% on test data
# log_name = '20190802-224937_e1000_b80_phon_common_utkarsh' # training loss = 0.37, val loss = 2.224, per =  12.8% on training data
# log_name = '20190724-110527_e100_b80_phon_bidir_utkarsh_CV' # Max validation = 32.6%, Fold 0 acc on test: 14.34%
# log_name = '20190719-234522_e2_b20_phon_bidir_utkarsh_CV' #2 epoch model
sequences = train_sequences[:495] # test_sequences # train_sequences[:247]
labels = train_labels[:495] # test_labels # train_labels[:247]
# class_names = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'UW', 'CH', 'D', 'G', 'HH', 'JH', 'K', 'L', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'Y', 'Z', '<start>', '<end>']
class_names = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'UH', 'UW', 'CH', 'D', 'DH', 'G', 'HH', 'JH', 'K', 'L', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'Y', 'Z', 'P', 'B', 'F', 'M', 'V', 'W', '<start>', '<end>']

# Loading the models
model = load_model('SavedModels/Full_{}.h5'.format(log_name))
encoder_model = load_model('SavedModels/Encoder_{}.h5'.format(log_name))
decoder_model = load_model('SavedModels/Decoder_{}.h5'.format(log_name))

# Loading files
# true_labels_file = open('ASRfiles/true_labels_test_{}.txt'.format(log_name), 'w')
# predicted_labels_file = open('ASRfiles/predicted_labels_test_{}.txt'.format(log_name), 'w')

# Calculating actual labels 
actual_labels = []
for label in np.argmax(labels, axis=-1).tolist():
    for idx, val in enumerate(label):
        if val==end_symbol:     actual_labels.append(label[:idx+1])


# Call the desired functins here
# get_predictions(true_labels_file, predicted_labels_file, sequences, labels, class_names, encoder_model, decoder_model,\
#  15, start_symbol, end_symbol, num_classes)
print
print actual_labels
print [list(x) for x in batch_greedy_decode(sequences, encoder_model, decoder_model, 15, start_symbol, end_symbol, num_classes)]
# for sequence in sequences:     
#     print beam_decode(sequence, encoder_model, decoder_model, 15, start_symbol, end_symbol, num_classes, k=10)
print error_rate(sequences, labels, encoder_model, decoder_model, 15, start_symbol, end_symbol, num_classes)
# print bit_rate(9,0.8,7)
# plot_confusion_matrix(sequences, labels, np.array(class_names),
#                       title='Confusion matrix, without normalization')
# # Plot normalized confusion matrix
# plot_confusion_matrix(sequences, labels, np.array(class_names), encoder_model, decoder_model, 15,\
#  start_symbol, end_symbol, num_classes, normalize=True, title='Normalized confusion matrix')
# plt.show()

