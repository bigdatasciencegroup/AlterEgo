import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
import os
import keyboard
from phue import Bridge
import logging
logging.basicConfig()

#b = Bridge('192.168.86.25')
#b.set_light('Hue color light 1', 'on', True)
#b.set_light('Hue color light 1', 'bri', 254)

import data

def control_tetris(index):
    if index == 0:
        keyboard.press_and_release('left')
        os.system('say left &')
    elif index == 1:
        keyboard.press_and_release('right')
        os.system('say right &')
    elif index == 2:
        keyboard.press_and_release('up')
        os.system('say rotate &')

def control_phue(index):
    if index == 0:
        b.set_light('Hue color light 1', 'hue', 0)
        os.system('say red &')
    elif index == 1:
        b.set_light('Hue color light 1', 'hue', 25500)
        os.system('say green &')
    elif index == 2:
        b.set_light('Hue color light 1', 'hue', 46920)
        os.system('say blue &')

classification_model = load_model('cnn_digits.h5')
activation_model = load_model('cnn_activation.h5')

num_classes = 10 #10

count = -1

class_window_size = 300 #300
act_window_size = 50 #25

history_size = 50
classification_history = [[0.0] * history_size for i in range(num_classes)]
activation_history = [0.0] * history_size
argmax_history = [0] * history_size
test_history = [0.0] * history_size
hypotheses = []
timestamp = 0
wait_time = 1.000
activated = False
activation_start = -1
last_prediction = None
prediction_timestamp = 0
prediction_time_limit = 5
def on_data(history):
    global count
    global output_history
    global hypotheses
    global timestamp
    global activated
    global activation_start
    global prediction_timestamp
    global last_prediction
    count += 1
    if count % 1 == 0:
#        classification_output = classification_model.predict(np.array([history[-class_window_size:,1:8]]))[0]
#        classification_output = classification_model.predict(np.array([history[-class_window_size:,4:7]]))[0]
        classification_output = classification_model.predict(np.array([history[-class_window_size:,4:8]]))[0]
        activation_output = activation_model.predict(
#            np.array(map(lambda i: history[-i-act_window_size:-i,1:8] if i > 0 else history[-act_window_size:,1:8],
#                         range(0, class_window_size, act_window_size))))
#            np.array(map(lambda i: history[-i-act_window_size:-i,4:7] if i > 0 else history[-act_window_size:,4:7],
#                         range(0, class_window_size, act_window_size))))
            np.array(map(lambda i: history[-i-act_window_size:-i,4:8] if i > 0 else history[-act_window_size:,4:8],
                         range(0, class_window_size, act_window_size))))
        activation_output = activation_output[:,1][::-1]
    
        argmax = np.argmax(classification_output)
        argmax_history.pop(0)
        argmax_history.append(argmax)
        
        print
        print
        print activation_output
            
        for i in range(len(activation_output)):
            n = int(50 * activation_output[i])
            print i, '[' + '#' * n + ' ' * (50 - n) + ']', int(activation_output[i] * 100), '%'
        print
        
#        activation_output = np.mean(activation_output)
#        activated = 1.0 if activation_output >= 0.98 else 0.0

        max_activation_output = 0.0
        consecutive = 5 #5
        for i in range(len(activation_output) - consecutive + 1):
#        for i in range(len(activation_output) - consecutive, len(activation_output) - consecutive + 1):
            max_activation_output = max(max_activation_output, np.mean(activation_output[i:i+consecutive]))
        if not activated and max_activation_output > 0.90:
            activated = True
            activation_start = -1
        if activated:
            activation_start -= 1
            if max_activation_output <= 0.85:
                activated = False
                now = time.time()
                if now - timestamp > wait_time:
                    prediction = argmax_history[max(-len(argmax_history), int((activation_start)/2))]
                    hypotheses.append(prediction)
                    timestamp = now
                    if last_prediction == prediction and now - prediction_timestamp <= prediction_time_limit:
#                        os.system('say ' + str(prediction) + ' &')
                        control_tetris(prediction)
#                        control_phue(prediction)
                    last_prediction = prediction
                    prediction_timestamp = now
                        
                
#        activated = True if max_activation_output > 0.98 else currently_activated
#        activated = False if max_activation_output <= 0.90 else currently_activated
        activation_output = max_activation_output
    
        test_history.pop(0)
        test_history.append(activation_output)
                
        for i in range(len(classification_output)):
            classification_history[i].pop(0)
            classification_history[i].append(classification_output[i])
        activation_history.pop(0)
        activation_history.append(activation_output)
            
        n = int(50 * activation_output)
        print 'activation', '[' + '#' * n + ' ' * (50 - n) + ']', int(activation_output * 100), '%'
        n = int(50 * activated)
        print ' activated', '[' + '#' * n + ' ' * (50 - n) + ']', int(activated * 100), '%'
        print
        for i in range(len(classification_output)):
            n = int(50 * classification_output[i])
            print i, '[' + '#' * n + ' ' * (50 - n) + ']', \
                    int(classification_output[i] * activation_output * 100), '%'
        print
        prob = classification_output[argmax]
        print argmax if prob > 0.5 else None, '\t', int(prob * 100), '%'
        print
        
#        max_hypothesis = (argmax, prob) if prob > max_hypothesis[1] else max_hypothesis
        
#        now = time.time()
#        if was_activated and not activated and now - timestamp > wait_time:
##            print max_hypothesis
#            hypotheses.append(max_hypothesis[0])
#            max_hypothesis = (None, 0.0)
#            timestamp = now
        print hypotheses
#        was_activated = activated
        

#columns = 4
#rows = int(math.ceil((num_classes+1)/float(columns)))
#fig = plt.figure(1, figsize=(4 * columns, 2 * rows))
#plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1, right=0.95, left=0.05)
#axis_matrix = fig.subplots(rows, columns)
#axes = reduce(lambda a,b: list(a)+list(b), axis_matrix) if columns > 1 and rows > 1 else axis_matrix
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


#data.from_file.start('data/data/8_subvocal_0_50_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/9_subvocal_1_50_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/13_subvocal_3_50_trials.txt', on_data, speed=1.0)

#data.from_file.start('data/data/39_subvocal_silence_100_trials.txt', on_data, speed=1.0)

#data.from_file.start('data/test_data/testing_silence_and_random_digits.txt', on_data, speed=1.0)
#data.from_file.start('data/test_data/testing_silence_and_some_threes_at_start.txt', on_data, speed=1.0)

#data.from_file.start('data/test_data/testing_1_2_3_5times_each.txt', on_data, speed=1.0)



#data.from_file.start('data/test_data/testing_0_s2.txt', on_data, speed=1.0)

#data.from_file.start('data/data/69_subvocal_1_x_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/59_subvocal_0_158_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/61_subvocal_3_193_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/62_subvocal_4_175_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/64_subvocal_6_166_trials.txt', on_data, speed=1.0)

#data.from_file.start('data/data/2_subvocal_digits_9_trials.txt', on_data, speed=1.0)


#data.from_file.start('data/test_data/testing_6_s3.txt', on_data, speed=1.0)

#data.from_file.start('data/test_data/testing_1_s4.txt', on_data, speed=1.0)

#data.from_file.start('data/data/91_subvocal_6_50_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/77_subvocal_2_50_trials.txt', on_data, speed=1.0)
#data.from_file.start('data/data/96_subvocal_silence_100_trials.txt', on_data, speed=1.0)


#data.from_file.start('data/test_data/testing_0_s7.txt', on_data, speed=1.0)

data.from_file.start('data/data/130_subvocal_1_50_trials.txt', on_data, speed=1.0)

