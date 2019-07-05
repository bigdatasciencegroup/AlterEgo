import numpy as np

sentences = [
    'because the american people want',
    'for this kind of program',
    'business would you start today',
    'something that we really have',
    'never leave home without it',
    'can not help but think',
    'to work with him again',
    'be back right after these',
    'let me say one other',
    'time in her life she',
    'how much do they know',
    'as a young child i',
    'go through all of them',
    'if you need a place',
    'i put the book down',
    'those of us who feel',
    'to make up my own',
    'i could see why he',
    'a way to give a',
    'study last year by the',
    'move from job to job',
    'to do a talk show',
    'us now in our new',
    'when you turn off the',
    'if i go into a',
    'should be as high as',
    'to be a very small',
    'the first day of each',
    'any case in which a',
    'in the head during a',
]

word_syllable_pairs = [(obj['word'], obj['syllables'])\
                       for obj in map(lambda x: eval(x.lower()),
                                      open('phoneme-groups-with-syllables.json', 'rb').readlines())]
word_syllable_map = {x[0]:map(tuple, x[1]) for x in word_syllable_pairs}

syllable_sets = []
for sentence in sentences:
    print sentence
    syllable_groups = map(word_syllable_map.get, sentence.split())
    syllable_sets.append(sum(syllable_groups, []))
    for syllables in syllable_groups:
        print syllables
    print syllable_sets[-1]
    print


all_syllables = np.unique(reduce(lambda a,b: a+b, syllable_sets, []))
syllable_map = {syllable:i for i, syllable in enumerate(all_syllables)}
print all_syllables
print len(set(all_syllables))

def get_syllable_counts(i):
#    contains, counts = np.unique(phoneme_sets[i], return_counts=True)
    contains_counts = {}
    for syllable in syllable_sets[i]:
        contains_counts[syllable] = contains_counts.get(syllable, 0) + 1
    contains = contains_counts.keys()
    counts = np.array(map(contains_counts.get, contains))
    syllable_counts = np.array([0] * len(syllable_map))
    for j, syllable in enumerate(contains):
        syllable_counts[syllable_map[syllable]] += counts[j]
    return syllable_counts

syllable_counts = map(get_syllable_counts, range(len(syllable_sets)))

print
for i in range(len(sentences)):
    print sentences[i]
    print list(syllable_counts[i])
print

print
print np.sum(syllable_counts, axis=0)