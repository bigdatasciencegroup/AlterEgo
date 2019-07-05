import numpy as np


process_fn = lambda x: tuple([int(x[0])]+x[1:])
gram_pairs = [[process_fn(l.strip().lower().split())\
               for l in open('coca_frequency/'+base+'.txt', 'rb').readlines()]\
              for base in ['w1_', 'w2_', 'w3_', 'w4_', 'w5_']]
gram_pairs = map(lambda y: sorted(y)[::-1], gram_pairs)
gram_pairs = map(lambda y: map(lambda x: (x[1:], x[0]), y), gram_pairs)

print map(len, gram_pairs)
print map(lambda y: sum(map(lambda x: x[1], y)), gram_pairs)

word_phonemes_pairs = [l.strip().lower().split() for l in open('cmudict.txt', 'rb').readlines()]
word_phoneme_map = {x[0]:x[1:] for x in word_phonemes_pairs}

word_syllable_pairs = [(obj['word'], obj['syllables'])\
                       for obj in map(lambda x: eval(x.lower()),
                                      open('phoneme-groups-with-syllables.json', 'rb').readlines())]
word_syllable_map = {x[0]:map(tuple, x[1]) for x in word_syllable_pairs}


phonemes = set(np.concatenate(word_phoneme_map.values()))
print phonemes
print len(phonemes)

#phonemes_subset = set(['sh', 'ih0', 'ih1', 'ih2', 't'])
#phonemes_subset = set(['ch', 'zh', 'uh0', 'uw2', 'uw1', 'uw0', 'ey2', 'aw2', 'aw1', 'aw0', 'uh2', 'ao2', 'ao1', 'uh1', 'ao0', 'ae1', 'ae0', 'ae2', 'er0', 'er1', 'er2', 'ey0', 'oy0', 'ng', 'r', 'aa1', 'iy0', 'th', 'iy2', 'iy1', 'aa0', 'ih1', 'dh', 'ih2', 'ih0', 'aa2', 'ah2', 'g'])
phonemes_subset = phonemes - set(['b', 'f', 'g', 'jh', 'm', 'p', 'r', 'v', 'w', 'uh0', 'aw0', 'ao2', 'oy0', 'er2', 'aa0'])

#print gram_pairs[0]
gram_pairs = map(lambda gp: filter(lambda x: all(map(lambda y: y in phonemes_subset,
                                                     np.concatenate(map(lambda w: word_phoneme_map.get(w, [None]), x[0])))), gp), gram_pairs)
print
print gram_pairs[0]
print
print gram_pairs[1]
print
print gram_pairs[2]
print
print gram_pairs[3]
print
print gram_pairs[4]

potential_sentences = map(lambda x: x[0], gram_pairs[4])
bigrams = set()
sentences = []
for sentence in potential_sentences:
    good = True
    for i in range(len(sentence)-1):
        if sentence[i:i+2] in bigrams:
            good = False
            break
    if good:
        sentences.append(sentence)
        for i in range(len(sentence)-1):
            bigrams.add(sentence[i:i+2])
        
print
print sentences
print len(sentences)

phoneme_counts = {ph: 0 for ph in phonemes_subset}
for sentence in sentences:
    for word in sentence:
        for phoneme in word_phoneme_map[word]:
            phoneme_counts[phoneme] += 1
print
print phoneme_counts
print phoneme_counts.keys()
print phoneme_counts.values()

print list(sorted(phoneme_counts.values()))

print
print map(lambda s: ' '.join(s), sentences)

#vocab = np.array([
#        'i',
#        'am',
#        'cold',
#        'hot',
#        'hungry',
#        'tired',
#        'want',
#        'need',
#        'food',
#        'water',
#        'hello',
#        'the',
#        'that',
#        'thank',
#        'you',
#        'where',
#        'what',
#        'of',
#        'because',
#        'feeling',
#    ])
#word_map = {word:i for i, word in enumerate(vocab)}
#
#np.random.seed(1)
#
#num_sentences = 25
#sentence_length = 5
#count = 0
#word_sets = []
#bigrams = set()
#for i in range(num_sentences):
#    while True:
#        count += 1
#    #    print count
#        words = vocab[np.random.choice(range(len(vocab)), sentence_length, replace=False)]
#        tmp = list(word_sets)
#        tmp.append(words)
#        contains, counts = np.unique(list(np.concatenate(tmp)) + list(vocab), return_counts=True)
#        counts -= 1
#        okay = max(counts) - min(counts) <= 2 and np.all(np.abs(counts - np.mean(counts)) <= 1.0)
#        for j in range(len(words)-2+1):
#            key = tuple(words[j:j+2])
#            if key in bigrams:
#                okay = False
#                break
#        if okay:
#            for j in range(len(words)-2+1):
#                key = tuple(words[j:j+2])
#                bigrams.add(key)
#            break
#    word_sets.append(words)
#    
#    print
#    print count
#    for words in word_sets:
#        print ' '.join(words)
#        print map(word_map.get, words)
#        
#    print
#    contains, counts = np.unique(np.concatenate(word_sets), return_counts=True)
#    print list(contains)
#    print list(counts)
#    