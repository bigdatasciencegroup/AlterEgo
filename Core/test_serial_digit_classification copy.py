import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time

import data

classification_model = load_model('cnn_digits.h5')
activation_model = load_model('cnn_activation.h5')

num_classes = 10

count = -1

class_window_size = 300
act_window_size = 25

history_size = 50
classification_history = [[0.0] * history_size for i in range(num_classes)]
activation_history = [0.0] * history_size
test_history = [0.0] * history_size
was_activated = False
max_hypothesis = (None, 0.0)
hypotheses = []
timestamp = 0
wait_time = 0.500
classifications = []
def on_data(history):
    global count
    global output_history
    global max_hypothesis
    global was_activated
    global hypotheses
    global timestamp
    count += 1
    if count % 1 == 0:
#        print time.time()

        classification_output = classification_model.predict(np.array([history[-class_window_size:,1:8]]))[0]
#        print time.time()
        
#        classification_output = np.ones((num_classes,))
    
        activation_output = activation_model.predict(
            np.array(map(lambda i: history[-i-act_window_size:-i,1:8] if i > 0 else history[-act_window_size:,1:8],
                         range(0, 300, act_window_size))))
        activation_output = activation_output[::-1]
        original_activations = np.array(activation_output)
        
#        activation_output = np.ones((12, 2))
        
        print activation_output
        
        for i in range(len(activation_output)):
            activation_output[i] = [1.0, 0.0] if activation_output[i][1] < 0.30 else [0.0, 1.0]
            
        activated = False
        on_count = 0
        off_count = 0
        for i in range(len(activation_output)):
            if activation_output[i][1]:
                on_count += 1
                if on_count > 0:
                    off_count = 0
#                if on_count >= 4 and on_count <= 6 and i == int(len(activation_output)/2.):
#                if on_count == 4 and i > 6 and i < 10: # good
#                if on_count == 4 and i > 6 and i < 8:
                if on_count == 6 and i > 6 and i < 10: # good
                    print 'TEST'
                    activated = True
                    break
            else:
                off_count += 1
                if off_count > 0:
                    on_count = 0
            
#        for i in range(len(activation_output)):
#            n = int(50 * activation_output[i][1])
#            print i, '[' + '#' * n + ' ' * (50 - n) + ']', int(activation_output[i][1] * 100), '%'
        print '[' + (''.join(['#' if x[1] else ' ' for x in activation_output])) + ']'
        print

        activation_output = [0.0, 1.0] if activated else [1.0, 0.0]
#        activation_output = np.mean(activation_output, axis=0)
#        activation_output = np.min(activation_output, axis=0)
#        activation_output = activation_output[0]
        test_history.pop(0)
        test_history.append(activation_output[1])
        
#        activation_output = [1.0, 0.0] if activation_output[1] < 0.75 else [0.0, 1.0]
        
        for i in range(len(classification_output)):
            classification_history[i].pop(0)
            classification_history[i].append(classification_output[i])
        activation_history.pop(0)
        activation_history.append(activation_output[1])
            
        n = int(50 * activation_output[1])
        print 'activation', '[' + '#' * n + ' ' * (50 - n) + ']', int(activation_output[1] * 100), '%'
        print
        for i in range(len(classification_output)):
#            n = int(50 * classification_output[i] * activation_output[1])
            n = int(50 * classification_output[i])
            print i, '[' + '#' * n + ' ' * (50 - n) + ']', \
                    int(classification_output[i] * activation_output[1] * 100), '%'
        print
        argmax = np.argmax(classification_output)
        prob = classification_output[argmax] * activation_output[1]
        print argmax if prob > 0.5 else None, '\t', int(prob * 100), '%'
        print
        
        max_hypothesis = (argmax, prob) if prob > max_hypothesis[1] else max_hypothesis
        
        if classification_output[argmax] > 0.9 and np.mean(original_activations[:,1]) > 0.5:
            classifications.append(argmax)
        
        now = time.time()
        if was_activated and not activated and now - timestamp > wait_time:
#            print max_hypothesis
            hypotheses.append(max_hypothesis[0])
            timestamp = now
            print hypotheses
            max_hypothesis = (None, 0.0)
        else:
#            print
            print hypotheses
        was_activated = activated
        
        print classifications
        

#columns = 4
#rows = int(math.ceil((num_classes+1)/float(columns)))
#fig = plt.figure(1, figsize=(4 * columns, 2 * rows))
#plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1, right=0.95, left=0.05)
#axis_matrix = fig.subplots(rows, columns)
#axes = reduce(lambda a,b: list(a)+list(b), axis_matrix) if columns > 1 else axis_matrix
#axes = axes[:num_classes+1]
#lines = [axes[i].plot([],[], lw=0.5)[0] for i in range(num_classes+1)]
#history_x = range(-history_size, 0)
#for i in range(num_classes):
#    axes[i].set_title('Class ' + str(i))
##    axes[i].set_xlabel('Timestep')
##    axes[i].set_ylabel('P(' + str(i) + '|X)')
#    axes[i].axis([-history_size, 0, 0.0, 1.0])
#axes[-1].axis([-history_size, 0, 0.0, 1.0])
#def update(i):
#    for j in range(num_classes):
#        lines[j].set_data(history_x, np.array(classification_history[j]) * activation_history)
#    lines[-1].set_data(history_x, test_history)
#    return lines
#
#line_ani = animation.FuncAnimation(fig, update, interval=100, blit=True)


data.serial.start('/dev/tty.usbserial-DQ007UBV', on_data, override_step=25)


