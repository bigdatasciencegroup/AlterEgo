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
all_phonemes = np.unique(np.concatenate(phoneme_sets))
phoneme_map = {phoneme:i for i, phoneme in enumerate(all_phonemes)}
print all_phonemes
print phoneme_map

print

avg_counts = []
phoneme_counts = []
for i, phonemes in enumerate(phoneme_sets):
    print stems[i]
    print sentences[i]
    print ' '.join(phonemes)
    contains, counts = np.unique(phonemes, return_counts=True)
    phoneme_counts.append(np.array([0] * len(phoneme_map)))
    for j, phoneme in enumerate(contains):
        phoneme_counts[i][phoneme_map[phoneme]] += counts[j]
    print phoneme_counts[i]
    avg_count = np.mean(phoneme_counts[i][np.where(phoneme_counts[i] > 0)])
    print len(phonemes), len(set(phonemes)), avg_count
    avg_counts.append((avg_count, i))
    print

print

test_set = sorted(avg_counts)[::-1]
test_indices = list(zip(*test_set)[1])
sentence_map = {}
for i in test_indices:
    if sentences[i] not in sentence_map:
        sentence_map[sentences[i]] = i
    if len(sentence_map) >= 10:
        break
selected_indices = sentence_map.values()
selected_sentences = sentence_map.keys()
for i, sentence in enumerate(selected_sentences):
    print i, stems[i]
    print sentence
    print ' '.join(phoneme_sets[i])
    print len(phoneme_sets[i]), len(set(phoneme_sets[i])), avg_counts[i][0]
    print
print

