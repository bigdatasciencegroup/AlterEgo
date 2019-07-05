import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import data

words = np.array(['0', '1', '2'])
filepaths = map(lambda word: 'vis2_012_test/visualized_' + word + '.npy', words)

visualized = map(np.load, filepaths)

visualized = data.transform.normalize_std(visualized)


def normalize_kernel(kernel, subtract_mean=False):
    if subtract_mean:
        kernel = np.array(kernel, np.float32) - np.mean(kernel)
    return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
sample_rate = 1000
avg_kernel = normalize_kernel([1.0] * (sample_rate // 20))
def deprocess(x):
    x = data.transform.correlate(x, avg_kernel)
    return x

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

#colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
ylims = (-2, 2)
for i in range(len(visualized)):
    for c in range(4):
        axes[i].plot(deprocess(visualized[i])[:,c])
#    axes[i].set_ylim(ylims)
plt.show()

