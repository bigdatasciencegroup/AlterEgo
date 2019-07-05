import numpy as np

labels = sum([[x for _ in range(10)] for x in range(15)], [])
#labels = sum([[x for _ in range(15)] for x in range(5)], [])
#labels = sum([[x for _ in range(30)] for x in range(4)], [])

np.random.seed(1)
np.random.shuffle(labels)

print 'Session 1'
print labels