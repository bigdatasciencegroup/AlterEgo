import numpy as np


process_fn = lambda x: tuple([int(x[0])]+x[1:])
gram_pairs = [[process_fn(l.strip().lower().split())\
               for l in open('coca_frequency/'+base+'.txt', 'rb').readlines()]\
              for base in ['w1_', 'w2_', 'w3_', 'w4_', 'w5_']]
gram_pairs = map(lambda y: sorted(y)[::-1], gram_pairs)
gram_pairs = map(lambda y: map(lambda x: (x[1:], x[0]), y), gram_pairs)

print map(len, gram_pairs)
print map(lambda y: sum(map(lambda x: x[1], y)), gram_pairs)

vocab = np.array([
        'i',
        'am',
        'cold',
        'hot',
        'hungry',
        'tired',
        'want',
        'need',
        'food',
        'water',
        'hello',
        'the',
        'that',
        'thank',
        'you',
        'where',
        'what',
        'of',
        'because',
        'feeling',
    ])
word_map = {word:i for i, word in enumerate(vocab)}

np.random.seed(1)

num_sentences = 200
sentence_length_range = range(4, 8)
count = 0
word_sets = []
bigrams = set()
for i in range(num_sentences):
    while True:
        sentence_length = np.random.choice(sentence_length_range)
        count += 1
    #    print count
        words = vocab[np.random.choice(range(len(vocab)), sentence_length, replace=False)]
        tmp = list(word_sets)
        tmp.append(words)
        contains, counts = np.unique(list(np.concatenate(tmp)) + list(vocab), return_counts=True)
        counts -= 1
        okay = max(counts) - min(counts) <= 2 and np.all(np.abs(counts - np.mean(counts)) <= 1.0)
        okay = 2 if okay else 0
        for j in range(len(words)-2+1):
            key = tuple(words[j:j+2])
            if key in bigrams:
                okay = max(0, okay - 1)
                break
        if okay:
            for j in range(len(words)-2+1):
                key = tuple(words[j:j+2])
                bigrams.add(key)
            break
    word_sets.append(words)
    
    print
    print count
    for words in word_sets:
        print ' '.join(words)
        print map(word_map.get, words)
        
    print
    contains, counts = np.unique(np.concatenate(word_sets), return_counts=True)
    print list(contains)
    print list(counts)
    print
    print str(len(word_sets)) + '/' + str(num_sentences)
    