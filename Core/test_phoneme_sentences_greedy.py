import numpy as np
import fnmatch
import os

root = 'timit/timit/train'
matches = []
for root, dirnames, filenames in os.walk(root):
    for filename in fnmatch.filter(filenames, 'si*.wav') + fnmatch.filter(filenames, 'sx*.wav'):
        matches.append(os.path.join(root, filename))
stems = map(lambda match: match.split('.')[0], matches)

phoneme_sets = [[x.strip().split()[-1] for x in open(stem + '.phn', 'rb').readlines()][1:-1] for stem in stems]
word_sets = [[x.strip().split()[-1] for x in open(stem + '.wrd', 'rb').readlines()] for stem in stems]
sentences = np.array(map(' '.join, word_sets))
print len(set(sentences))

unique_map = {}
for i in range(len(sentences)):
    if sentences[i] not in unique_map:
        unique_map[sentences[i]] = i
unique_indices = unique_map.values()

stems = np.array(stems)[unique_indices]
phoneme_sets = np.array(phoneme_sets)[unique_indices]
word_sets = np.array(word_sets)[unique_indices]
sentences = np.array(sentences)[unique_indices]

all_phonemes = np.unique(np.concatenate(phoneme_sets))
phoneme_map = {phoneme:i for i, phoneme in enumerate(all_phonemes)}
print all_phonemes
#print phoneme_map
print len(set(all_phonemes))

phoneme_counts = []
for i, phonemes in enumerate(phoneme_sets):
    contains, counts = np.unique(phonemes, return_counts=True)
    phoneme_counts.append(np.array([0] * len(phoneme_map)))
    for j, phoneme in enumerate(contains):
        phoneme_counts[i][phoneme_map[phoneme]] += counts[j]

num_phonemes_pairs = map(lambda (i, x): (sum(x / x), i), enumerate(phoneme_counts))
avg_count_pairs = map(lambda (i, x): (np.mean(x[np.where(x > 0)]), i), enumerate(phoneme_counts))
selected_index = sorted(num_phonemes_pairs)[1][1]
#selected_index = sorted(avg_count_pairs)[0][1]

start_index = selected_index
num_sentences = 5
indices = [start_index]
current_phoneme_counts = phoneme_counts[start_index]
for _ in range(1, num_sentences):
    positive_mask = current_phoneme_counts > 0
    values = []
    for i in range(len(sentences)):
        if i not in indices:
    #        print list(current_phoneme_counts)
    #        print list(phoneme_counts[i])
    #        print list(positive_mask * 2 - 1)
    #        print list(phoneme_counts[i] * (positive_mask * 2 - 1))
            value = sum(phoneme_counts[i] * (positive_mask * 2 - 1))
            positives = sum(phoneme_counts[i] * (positive_mask * 1))
            negatives = sum(phoneme_counts[i] * (positive_mask * 1 - 1))
#            values.append((value, negatives, positives, i))
            values.append((negatives, positives, value, i))
    print
    for i in list(indices):
        print sentences[i]
    print
    ordered = sorted(values)[::-1]
    for value in ordered[:5]:
        print value, sentences[value[3]]
    indices.append(ordered[0][3])
    current_phoneme_counts += phoneme_counts[ordered[0][3]]
    print list(current_phoneme_counts)

print
selected_phonemes = np.unique(np.concatenate(phoneme_sets[indices]))
selected_phoneme_map = {phoneme:i for i, phoneme in enumerate(selected_phonemes)}
print selected_phonemes
for i in list(indices):
    print sentences[i]
    print map(selected_phoneme_map.get, phoneme_sets[i])
print
print len(set(np.concatenate(phoneme_sets[indices])))
#print list(current_phoneme_counts[map(phoneme_map.get, selected_phonemes)])
print list(current_phoneme_counts[map(phoneme_map.get, selected_phonemes)])
