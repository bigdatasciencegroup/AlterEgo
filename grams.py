from nltk.util import ngrams

tokens = ['a', 'b', 'c', 'd']
output = list(ngrams(tokens,2))

print(output)