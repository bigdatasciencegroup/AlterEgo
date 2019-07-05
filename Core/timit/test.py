import numpy as np
import fnmatch
import os
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re

def process_data(root, vocabulary_map=None, use_phonemes=True):
    sequences = []
    labels = []
    matches = []
    print 'Loading metadata from', root
    for root, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames, 'si*.wav') + fnmatch.filter(filenames, 'sx*.wav'):
            matches.append(os.path.join(root, filename))
    stems = map(lambda match: match.split('.')[0], matches)
    for i in range(len(stems)):
        if (i+1) % 1 == 0: print str(i+1) + '/' + str(len(stems)), \
                                '(' + str(int((i+1)*100/len(stems))) + '%)', stems[i]
        data, _ = sf.read(stems[i] + '.wav')
        if use_phonemes:
            lines = map(lambda line: line.strip().split(' '), open(stems[i] + '.phn', 'r').readlines())[1:-1]
        else:
            lines = map(lambda line: line.strip().split(' '), open(stems[i] + '.wrd', 'r').readlines())
        label_spans = map(lambda x: tuple(map(int, x[:-1]) + x[-1:]), lines)
        mfcc = librosa.feature.mfcc(data)
        mfcc = list((mfcc - np.mean(mfcc)) / np.std(mfcc))
        label_sequence = map(lambda x: x[2], label_spans)
        sequences.append(mfcc)
        labels.append(label_sequence)
    if vocabulary_map is None:
        unique_labels = np.unique(sum(labels, []))
        print unique_labels
        print len(unique_labels)
        vocabulary_map = {unique_labels[i]:i for i in range(len(unique_labels))}
        print vocabulary_map
    else:
        labels, sequences = map(list, zip(*filter(lambda (label, _): all(map(lambda x: x in vocabulary_map, label)),
                                                  zip(labels, sequences))))
    labels = map(lambda seq: map(lambda x: vocabulary_map[x], seq), labels)
    return sequences, labels, vocabulary_map
    
train_sequences, train_labels, vocabulary_map = process_data('timit/train', use_phonemes=True)
test_sequences, test_labels, _ = process_data('timit/test', vocabulary_map, use_phonemes=True)
    
print np.shape(train_sequences)
print np.shape(train_labels)
print np.shape(test_sequences)
print np.shape(test_labels)
    
print
for i in range(len(train_sequences)):
    print str(i+1) + '/' + str(len(train_sequences)), '(' + str(int((i+1)*100/len(train_sequences))) + '%)' 
    np.save('tensorflow_CTC_example/train_data/data/' + str(i) + '.npy', train_sequences[i])
    np.save('tensorflow_CTC_example/train_data/labels/' + str(i) + '.npy', train_labels[i])
    
print
for i in range(len(test_sequences)):
    print str(i+1) + '/' + str(len(test_sequences)), '(' + str(int((i+1)*100/len(test_sequences))) + '%)' 
    np.save('tensorflow_CTC_example/test_data/data/' + str(i) + '.npy', test_sequences[i])
    np.save('tensorflow_CTC_example/test_data/labels/' + str(i) + '.npy', test_labels[i])




#tmp = sf.read('train/dr1/fcjf0/sa1.wav')[0]
#plt.plot(tmp)
#plt.show()

#tmp = librosa.feature.mfcc(tmp)
#print np.shape(tmp)
#plt.plot(tmp)
#plt.show()

