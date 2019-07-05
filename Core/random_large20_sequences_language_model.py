import numpy as np
import os

process_fn = lambda x: tuple([int(x[0])]+x[1:])
gram_pairs = [[process_fn(l.strip().lower().split())\
               for l in open('coca_frequency/'+base+'.txt', 'rb').readlines()]\
              for base in ['w1_', 'w2_', 'w3_', 'w4_', 'w5_']]
gram_pairs = map(lambda y: sorted(y)[::-1], gram_pairs)
gram_pairs = map(lambda y: map(lambda x: (x[1:], x[0]), y), gram_pairs)

gram_maps = map(lambda y: {x[0]:x[1] for x in y}, gram_pairs)

print map(len, gram_pairs)
print map(lambda y: sum(map(lambda x: x[1], y)), gram_pairs)

most_common_words = map(lambda x: x[0][0], gram_pairs[0][:100])
print most_common_words

#vocab = np.array(most_common_words)

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

vocab = np.array([
        'i',
        'am',
        'you',
        'are',
        'the',
        'want',
        'need',
        'cold',
        'hot',
        'food',
        'where',
        'what',
        'how',
        'feeling',
        'doing',
        'tired',
        'water',
        'hungry',
        'thirsty',
        'hello',
        
#        'that',
#        'probably',
#        'thank',
#        'of',
#        'because',
    ])

word_map = {word:i for i, word in enumerate(vocab)}

np.random.seed(1)

num_sentences = 200 #25
sentence_length_range = range(3, 7)
sentence_lengths = []
for i in range(num_sentences):
    sentence_lengths.append(int(float(i)/num_sentences*len(sentence_length_range)) + min(sentence_length_range))
print sentence_lengths
count = 0
word_sets = []
sentences = []
label_sequences = []
bigrams = {}
trigrams = {}
for i in range(num_sentences):
    while True:
#        sentence_length = np.random.choice(sentence_length_range)
        sentence_length = sentence_lengths[i]
    #    print count
        pool = []
#        degree = min(sentence_length, 4) #3
        degree = 2 #3
        for j in range(30): #20
            degree_count = 0
            while True:
                count += 1
                degree_count += 1
                words = vocab[np.random.choice(range(len(vocab)), sentence_length, replace=False)]
                tmp = list(word_sets)
                tmp.append(words)
                contains, counts = np.unique(list(np.concatenate(tmp)) + list(vocab), return_counts=True)
                counts -= 1
#                print words
#                okay = max(counts) - min(counts) <= 2 and np.all(np.abs(counts - np.mean(counts)) <= 1.0)
#                okay = max(counts) - min(counts) <= 3 # REMOVE
#                okay = max(counts) - min(counts) <= 3 and np.all(np.abs(counts - np.mean(counts)) <= 2.0) # REMOVE
#                okay = max(counts) - min(counts) <= 2 # REMOVE
                okay = max(counts) - min(counts) <= 4
#                okay = True # REMOVE
                okay = 20 if okay else 0
                okay = okay if tuple(words) not in set(map(tuple, word_sets)) else 0
                for j in range(len(words)-2+1):
                    key = tuple(words[j:j+2])
                    if key in trigrams:
                        okay = max(0, okay - trigrams[key] * 2)
                    if key in bigrams:
                        okay = max(0, okay - bigrams[key])
                if not okay:
                    continue
                log_probs = []
                for n in range(1, len(gram_pairs)):
    #            for n in range(1, 2):
                    log_prob = 0.0
                    for k in range(len(words)-n):
                        gram = tuple(words[k:k+n+1])
                        dependent = tuple(words[k:k+n])
                        num, denom = gram_maps[n].get(gram, 0), max(1, gram_maps[n-1].get(dependent, 0))
                        prob_update = min(float(num) / denom, 1.0)
    #                    print gram, prob_update
#                        log_prob += np.log(prob_update)
                        log_prob += max(np.log(1e-18), np.log(prob_update))
                    log_prob += np.log(len(words))
                    log_probs.append(log_prob)
#                if np.all((np.array(log_probs) > -np.inf)[:degree]):
                if np.all((np.array(log_probs) > np.log(1e-18))[:degree]):
                    pool.append(tuple(log_probs[::-1] + [list(words)]))
                    degree_count = 0
                    print pool[-1]
                    break
                else:
                    if degree_count > 10000: #20000 * 3 ** (degree):
                        degree -= 1
                        degree_count = 0
                        print 'Reducing constraints'
        print
        pool = sorted(pool)
        for pair in pool:
            print pair
        words = np.array(pool[-1][-1])
        for j in range(len(words)-2+1):
            key = tuple(words[j:j+2])
#            bigrams.add(key)
            bigrams[key] = bigrams.get(key, 0) + 1
        for j in range(len(words)-3+1):
            key = tuple(words[j:j+3])
#            bigrams.add(key)
            trigrams[key] = trigrams.get(key, 0) + 1
        break
    word_sets.append(words)
    os.system('say "#' + str(len(word_sets)) + '. ' + ' '.join(words) + '" &')
    
    print
    print count
    for words in word_sets:
        print '"' + ' '.join(words) + '"'
        print map(word_map.get, words)
        
    print
    contains, counts = np.unique(list(np.concatenate(word_sets)) + list(vocab), return_counts=True)
    counts -= 1
    print list(contains)
    print list(counts)
    print
    print str(len(word_sets)) + '/' + str(num_sentences)

print

for words in word_sets:
    sentences.append(' '.join(words))
    label_sequences.append(map(word_map.get, words))
    print sentences[-1]
    print label_sequences[-1]
    
print

print sentences
print
print label_sequences
print
    