import numpy as np

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
    ])
word_map = {word:i for i, word in enumerate(vocab)}

np.random.seed(1)

num_sentences = 25
sentence_length = 5
count = 0
word_sets = []
bigrams = set()
for i in range(num_sentences):
    while True:
        count += 1
    #    print count
        words = vocab[np.random.choice(range(len(vocab)), sentence_length, replace=False)]
        tmp = list(word_sets)
        tmp.append(words)
        contains, counts = np.unique(np.concatenate(tmp), return_counts=True)
        okay = max(counts) - min(counts) <= 2 and np.all(np.abs(counts - np.mean(counts)) <= 1.0)
        for j in range(len(words)-2+1):
            key = tuple(words[j:j+2])
            if key in bigrams:
                okay = False
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
    