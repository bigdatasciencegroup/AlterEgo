import numpy as np


word_phonemes_pairs = [l.strip().lower().split() for l in open('cmudict.txt', 'rb').readlines()]
word_phoneme_map = {x[0]:x[1:] for x in word_phonemes_pairs}

process_fn = lambda x: tuple([int(x[0])]+x[1:])
gram_pairs = [[process_fn(l.strip().lower().split())\
               for l in open('coca_frequency/'+base+'.txt', 'rb').readlines()]\
              for base in ['w1_', 'w2_', 'w3_', 'w4_', 'w5_']]
gram_pairs = map(lambda y: sorted(y)[::-1], gram_pairs)

gram_pairs[0] = gram_pairs[0][:250]

gram_pairs = map(lambda y: map(lambda x: (x[1:], x[0]), y), gram_pairs)

print map(len, gram_pairs)
words = map(lambda x: x[0][0], gram_pairs[0])
#print len(set(words))
words = set(filter(lambda w: w in word_phoneme_map, words))
#print len(words)
gram_pairs = map(lambda y: filter(lambda x: all([z in words for z in x[0]]), y), gram_pairs)
print map(len, gram_pairs)

#gram_counts = map(lambda y: {x[0]:x[1] for x in y}, gram_pairs)


#phoneme_sets = [x[1:] for x in word_phonemes_pairs]
#word_sets = [x[0:1] for x in word_phonemes_pairs]
#word_sets = map(lambda x: x[0], gram_pairs[2] + gram_pairs[3] + gram_pairs[4])
#word_sets = map(lambda x: x[0], gram_pairs[3] + gram_pairs[4])
word_sets = map(lambda x: x[0], gram_pairs[4])
phoneme_sets = map(lambda y: sum(map(word_phoneme_map.get, y), []), word_sets)
sentences = np.array(map(' '.join, word_sets))
print len(set(sentences))

#1/0

# If the sentences are not guaranteed to be unique
#unique_map = {}
#for i in range(len(sentences)):
#    if sentences[i] not in unique_map:
#        unique_map[sentences[i]] = i
#unique_indices = unique_map.values()
#
##stems = np.array(stems)[unique_indices]
#phoneme_sets = np.array(phoneme_sets)[unique_indices]
#word_sets = np.array(word_sets)[unique_indices]
#sentences = np.array(sentences)[unique_indices]

word_sets = np.array(word_sets)
phoneme_sets = np.array(phoneme_sets)
sentences = np.array(sentences)

all_phonemes = np.unique(np.concatenate(phoneme_sets))
phoneme_map = {phoneme:i for i, phoneme in enumerate(all_phonemes)}
print all_phonemes
#print phoneme_map
print len(set(all_phonemes))

word_phoneme_count_map = {}
for word in words:
    phoneme_counts = np.array([0] * len(phoneme_map))
    contains, counts = np.unique(word_phoneme_map[word], return_counts=True)
    for i, phoneme in enumerate(contains):
        phoneme_counts[phoneme_map[phoneme]] += counts[i]
    word_phoneme_count_map[word] = phoneme_counts

sentence_diversities = [0.0] * len(word_sets)
for i, word_set in enumerate(word_sets):
    word_level_phoneme_counts = []
    for j, word1 in enumerate(word_set):
        for k, word2 in enumerate(word_set):
            if j != k:
                sentence_diversities[i] += np.linalg.norm(
                    word_phoneme_count_map[word1] - word_phoneme_count_map[word2])
#    sentence_diversities[i] /= len(phoneme_sets[i])
    sentence_diversities[i] /= len(word_set)

def get_phoneme_counts(i):
    contains, counts = np.unique(phoneme_sets[i], return_counts=True)
    phoneme_counts = np.array([0] * len(phoneme_map))
    for j, phoneme in enumerate(contains):
        phoneme_counts[phoneme_map[phoneme]] += counts[j]
    return phoneme_counts

phoneme_counts = map(get_phoneme_counts, range(len(phoneme_sets)))
        
phoneme_ngrams = [set() for _ in range(2)]
for n in range(len(phoneme_ngrams)):
    for phoneme_set in phoneme_sets:
        for i in range(len(phoneme_set)-n-1):
            phoneme_ngrams[n].add(tuple(phoneme_set[i:i+n+2]))
    phoneme_ngrams[n] = list(phoneme_ngrams[n])
    print len(phoneme_ngrams[n])
    
ngram_maps = [{} for _ in range(len(phoneme_ngrams))]
for n in range(len(phoneme_ngrams)):
    for i, ngram in enumerate(phoneme_ngrams[n]):
        ngram_maps[n][ngram] = i
        
def get_ngram_counts(i):
    ngram_counts = []
    for n in range(len(phoneme_ngrams)):
        ngram_counts.append(np.array([0] * len(phoneme_ngrams[n])))
        for j in range(len(phoneme_sets[i])-n-1):
            ngram_counts[n][ngram_maps[n][tuple(phoneme_sets[i][j:j+n+2])]] += 1
    return ngram_counts

#1/0

num_phonemes_pairs = map(lambda (i, x): (sum(x / x), i), enumerate(phoneme_counts))
avg_count_pairs = map(lambda (i, x): (np.mean(x[np.where(x > 0)]), i), enumerate(phoneme_counts))
diversity_pairs = map(lambda (i, x): (x, i), enumerate(sentence_diversities))

#selected_index = sorted(num_phonemes_pairs)[1][1]
#selected_index = sorted(avg_count_pairs)[::1][0][1]
selected_index = sorted(diversity_pairs)[::-1][0][1]

#n1 = get_ngram_counts(112705)
#n2 = get_ngram_counts(112693)
#print list(n1[0])
#print list(n2[0])
#print np.sum(np.min([n1[0], n2[0]], axis=0))

start_index = selected_index
num_sentences = 50
indices = [start_index]
current_phoneme_counts = phoneme_counts[start_index]
current_ngram_counts = get_ngram_counts(start_index)
for _ in range(1, num_sentences):
    positive_mask = current_phoneme_counts > 0
    values = []
    for i in range(len(sentences)):
        if (i+1) % 1000 == 0: print str(i+1) + '/' + str(len(sentences))
        if i not in indices:
            ngram_counts = get_ngram_counts(i)
            ngram_neg_sum = -sum([np.sum(np.min([current_ngram_counts[k], ngram_counts[k]], axis=0))\
                                  for k in range(len(ngram_counts))])
#            value = sum(phoneme_counts[i] * (positive_mask * 2 - 1))
#            positives = sum(phoneme_counts[i] * (positive_mask * 1))
#            negatives = sum(phoneme_counts[i] * (positive_mask * 1 - 1))
            diversity = sentence_diversities[i]
#            values.append((value, ngram_neg_sum, negatives, diversity, positives, i))
#            values.append((ngram_neg_sum, negatives, diversity + positives, positives, value, i))
#            values.append((ngram_neg_sum + positives + diversity, negatives, i))
            values.append((5 * ngram_neg_sum + diversity, i))
    print
    for i in list(indices):
        print i, sentences[i]
    print
    ordered = sorted(values)[::-1]
    for value in ordered[:5]:
        print value, sentences[value[-1]]
    indices.append(ordered[0][-1])
    current_phoneme_counts += phoneme_counts[ordered[0][-1]]
    new_ngram_counts = get_ngram_counts(ordered[0][-1])
    for i in range(len(current_ngram_counts)):
        current_ngram_counts[i] += new_ngram_counts[i]
    print list(current_phoneme_counts)

print
selected_phonemes = np.unique(np.concatenate(phoneme_sets[indices]))
selected_phoneme_map = {phoneme:i for i, phoneme in enumerate(selected_phonemes)}
print selected_phonemes
print
for i in list(indices):
    print sentences[i]
    print map(selected_phoneme_map.get, phoneme_sets[i])
print
print len(set(np.concatenate(phoneme_sets[indices])))
#print list(current_phoneme_counts[map(phoneme_map.get, selected_phonemes)])
print list(current_phoneme_counts[map(phoneme_map.get, selected_phonemes)])
