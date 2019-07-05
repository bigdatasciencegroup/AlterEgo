import numpy as np

labels = sum([[x for _ in range(15)] for x in range(10)], [])


np.random.seed(1)
np.random.shuffle(labels)

print 'Session 1'
print labels


np.random.seed(2)
np.random.shuffle(labels)

print 'Session 2'
print labels