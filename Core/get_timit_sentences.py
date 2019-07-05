import numpy as np
import os
import fnmatch
import re

word_phonemes_pairs = [l.strip().lower().split() for l in open('cmudict.txt', 'rb').readlines()]
word_phoneme_map = {x[0]:x[1:] for x in word_phonemes_pairs}

word_syllable_pairs = [(obj['word'], obj['syllables'])\
                       for obj in map(lambda x: eval(x.lower()),
                                      open('phoneme-groups-with-syllables.json', 'rb').readlines())]
word_syllable_map = {x[0]:map(tuple, x[1]) for x in word_syllable_pairs}


root = 'timit/timit/train'
matches = []
print 'Loading metadata from', root
for root, dirnames, filenames in os.walk(root):
    for filename in fnmatch.filter(filenames, 'si*.txt'):# + fnmatch.filter(filenames, 'sx*.txt'):
        matches.append(os.path.join(root, filename))
        
sentences = map(lambda fp: re.sub(r'[^\w\s]', '', ' '.join(open(fp, 'r').readlines()[0].strip().lower().split()[2:])), matches)
for sentence in sentences:
    print sentence
print len(sentences)

#print
#print sentences[:300]

print
syllable_sets = []
for sentence in sentences:
    word_set = sentence.split(' ')
    syllable_set = map(lambda x: word_syllable_map.get(x, None), word_set)
    syllable_sets.append(syllable_set)
    print syllable_set
print
#syllable_set_lengths = map(lambda x: sum(x, []), syllable_sets)

#sentence_syllable_pairs = zip(syllable_set_lengths, sentences)
sentence_syllable_pairs = []
for i in range(len(sentences)):
    if None not in syllable_sets[i]:
        sentence_syllable_pairs.append((len(sum(syllable_sets[i], [])), len(sum(map(lambda x: sum(map(list, x), []), syllable_sets[i]), [])), syllable_sets[i], sentences[i]))
sentence_syllable_pairs = list(sorted(sentence_syllable_pairs))

print
for sentence_syllable_pair in sentence_syllable_pairs:
    print sentence_syllable_pair[0], sentence_syllable_pair[3]
    
sentence_syllable_pairs = sentence_syllable_pairs[:400]

np.random.seed(1)
np.random.shuffle(sentence_syllable_pairs)
    
syllable_counts, phoneme_counts, syllable_sets, sentences = zip(*sentence_syllable_pairs)
print
print syllable_counts
print phoneme_counts
print syllable_sets
print list(sentences)

print
print

print set(sorted(sum(map(list, sum(sum(syllable_sets, []), [])), [])))
print set(sorted(sum(map(list, sum(sum(syllable_sets[100:], []), [])), [])))
print set(sorted(sum(map(list, sum(sum(syllable_sets[:100], []), [])), [])))
print set(sorted(sum(map(list, sum(sum(syllable_sets[100:], []), [])), []))) - set(sorted(sum(map(list, sum(sum(syllable_sets[:100], []), [])), [])))
print
print np.unique(sum(map(list, sum(sum(syllable_sets, []), [])), []), return_counts=True)
print np.unique(sum(map(list, sum(sum(syllable_sets[100:], []), [])), []), return_counts=True)
print np.unique(sum(map(list, sum(sum(syllable_sets[:100], []), [])), []), return_counts=True)

phonemes = list(set(sorted(sum(map(list, sum(sum(syllable_sets, []), [])), []))))
print phonemes
phoneme_map = {}
for i, phoneme in enumerate(phonemes):
    phoneme_map[phoneme] = i
print
print phoneme_map
print
print map(lambda syllable_set: map(lambda x: phoneme_map[x], sum(map(list, sum(syllable_set, [])), [])), syllable_sets)


