import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

import data

classification_model = load_model('cnn_words_16.h5')
activation_model = load_model('cnn_activation.h5')

class_map = ['the', 'a', 'is', 'it', 'what', 'where', 'time', 'year', 'day', 'plus', 'minus', 'about', 'student', 'government', 'important', 'information']

num_classes = 16

count = -1
history_size = 300
classification_history = [[0.0] * history_size for i in range(num_classes)]
activation_history = [0.0] * history_size
def on_data(history):
    global count
    global output_history
    count += 1
    if count % 1 == 0:
        classification_output = classification_model.predict(np.array([history[-300:,1:8]]))[0]
        activation_output = activation_model.predict(np.array([history[-300:,1:8]]))[0]
        for i in range(len(classification_output)):
            classification_history[i].pop(0)
            classification_history[i].append(classification_output[i])
        activation_history.pop(0)
        activation_history.append(activation_output[1])
            
        n = int(50 * activation_output[1])
        print 'activation', '[' + '#' * n + ' ' * (50 - n) + ']', int(activation_output[1] * 100), '%'
        print
        for i in range(len(classification_output)):
            n = int(50 * classification_output[i] * activation_output[1])
            print i, '[' + '#' * n + ' ' * (50 - n) + ']', \
                    int(classification_output[i] * activation_output[1] * 100), '%'
        print
        argmax = np.argmax(classification_output)
        print argmax, '', int(classification_output[argmax] * activation_output[1] * 100), '%'
        print
        

columns = 4
rows = int(math.ceil(num_classes/float(columns)))
fig = plt.figure(1, figsize=(4 * columns, 2 * rows))
plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1, right=0.95, left=0.05)
axis_matrix = fig.subplots(rows, columns)
axes = reduce(lambda a,b: list(a)+list(b), axis_matrix) if columns > 1 else axis_matrix
axes = axes[:num_classes]
lines = [axes[i].plot([],[], lw=0.5)[0] for i in range(num_classes)]
history_x = range(-history_size, 0)
for i in range(num_classes):
    axes[i].set_title('Class ' + class_map[i])
#    axes[i].set_xlabel('Timestep')
#    axes[i].set_ylabel('P(' + str(i) + '|X)')
    axes[i].axis([-history_size, 0, 0.0, 1.0])
def update(i):
    for j in range(len(classification_history)):
        lines[j].set_data(history_x, np.array(classification_history[j]) * activation_history)
    return lines

line_ani = animation.FuncAnimation(fig, update, interval=100, blit=True)


data.serial.start('/dev/tty.usbserial-DQ007UBV', on_data)


