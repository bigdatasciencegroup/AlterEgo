import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

words = np.array(['i', 'am', 'cold', 'hot', 'hungry', 'tired', 'want', 'need', 'food', 'water'])
filepaths = map(lambda word: 'vis40_words10/visualized_' + word + '.npy', words)

visualized = map(np.load, filepaths)


def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
sample_rate = 1000
avg_kernel = normalize_kernel([1.0] * (sample_rate // 20))
def deprocess(x):
    x = data.transform.correlate(x, avg_kernel)
    return x

fig, axes = plt.subplots(2, 5, figsize=(16, 6))

colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
for i in range(len(visualized)):
    x, y = divmod(i, 5)
    for c in range(8):
        axes[x][y].plot(deprocess(visualized[i])[:,c], c=colors[c])
plt.show()

