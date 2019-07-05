import numpy as np

def num_vowels(s):
    vowels = set(['a', 'e', 'i', 'o', 'u'])
    return sum(map(lambda c: c in vowels, s))

def num_syllables(s):
    vowels = set(['a', 'e', 'i', 'o', 'u'])
    vowels_y = vowels | set(['y'])
    return sum(map(lambda i: (i-1 < 0 or s[i-1] not in vowels) and s[i] in (vowels_y if i>0 else vowels) and (i < len(s)-1 or s[i] not in ['e']), range(len(s))))

lines = filter(lambda x: x, map(lambda x: x.lower().strip(), open('most_common_words.csv', 'r').readlines()))
words = np.array(map(lambda x: x.split(',')[1:3], lines))
print len(words)
print
print 'Top 30 any'
print list(words[:30,0])
print
print 'Top 30 nouns'
print list(words[np.where(words[:,1] == 'n')][:30,0])
print
print 'Top 30 verbs'
print list(words[np.where(words[:,1] == 'v')][:30,0])
print
print 'Top 30 adjectives'
print list(words[np.where(words[:,1] == 'j')][:30,0])
print
for i in range(5):
    print
    print i+1, 'syllable' + ('s' if i+1 != 1 else '')
    print list(words[np.where(np.array(map(num_syllables, words[:,0])) == i+1)][:30,0])
print