import numpy as np
import serial
import binascii
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
from scipy.signal import butter, iirnotch, lfilter
from scipy.ndimage.filters import gaussian_filter1d

import transform

#ADS1299_Vref = 4.5  #reference voltage for ADC in ADS1299.  set by its hardware
#ADS1299_gain = 24.0  #assumed gain setting for ADS1299.  set by its Arduino code
#scale_factor = ADS1299_Vref/float((pow(2,23)-1))/ADS1299_gain*1000000.

elapseds = [0.0] * 10
timestamp = time.time()

history_size = 1800
shown_size = 600

fs = 250

step = 35 #10 #25
#step = 15 #10 #25
count = 0
finished = False
recorded_count = 0

sample_count = 0
    
def start(filepath, callback_fn, channels=range(0, 8), history_size=history_size, shown_size=shown_size,
          transform_fn=transform.default_transform, plot=True, return_transformed=True,
          override_step=None, speed=1.0, **kwargs):
    
    history = [[0.0] * 8 for i in range(history_size)]
    trigger_history = [0.0] * history_size
    index_history = [0] * history_size
    global step
    if override_step:
        step = override_step
        
    contents = map(lambda x: x.strip(), open(filepath, 'r').readlines())
    frames = filter(lambda x: x and x[0] != '%', contents)
#    frames = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames)[starting_point:]
    frames = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames)

    fig = plt.figure(0, figsize=(12, 5))
    if plot:
        colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
        ax = fig.gca()
        trigger_line = ax.plot([-shown_size, 0], [0, 0], lw=1.0, c='black', alpha=0.5)[0]
        lines = np.array(
            [(ax.plot([],[], '-', lw=1.5, c=colors[i])[0] if i in channels else None) for i in range(8)])
        #line = ax.plot([],[], '-', lw=0.5, c=colors[1])[0]
        ax.set_title('Serial Data')
        ax.set_xlabel('Sample')
        if kwargs.get('apply_subtract_mean', transform.apply_subtract_mean) \
            or kwargs.get('apply_bandpass_filter', transform.apply_bandpass_filter):
            ax.axis([-shown_size, 0, -187500/8/128, 187500/8/128])
        else:
#            ax.axis([-history_size, 0, 0, 187500*2])
            ax.axis([-shown_size, 0, -187500, 187500])
    
    
        infos = [ax.text(0.005, 0.96 - 0.05*i, ' Channel ' + str(i+1) + ' ',
                         color=('black' if i in channels else 'white'),
                         bbox={'facecolor': (colors[i] if i in channels else 'black'),
                               'alpha':0.5 if i in channels else 0.2, 'pad':1},
                         transform=ax.transAxes, ha='left')\
                 for i in range(8)]
        
        ax.get_yaxis().set_ticks([])
        
        plt.subplots_adjust(left=0.04, right=0.96, bottom=0.09, top=0.92)

        history_x = range(-shown_size, 0)

    def update(i):
        global count
        global elapseds
        global timestamp
        global finished
        global recorded_count
        global sample_count
        
        if finished: return lines if plot else []
        
        count += 1
        
        elapsed = time.time() - timestamp
        elapseds.pop(0)
        elapseds.append(elapsed)
        timestamp = time.time()
#        print float(step) / fs - np.mean(elapseds)
        time.sleep(max(0, float(step) / (fs * speed) - np.mean(elapseds)))
#        print count * step, '/', len(frames)
        k = step
        if len(frames) - count * step <= k:
            k = len(frames) - count * step
            finished = True
        for l in range(k):
            sample_count += 1
            sample = frames[count * step + l]
            index = int(sample[0])
            
#            channel_data = [0.0] * 8
#            for i in range(8):
#                channel_data[i] = float(sample[i+1]) # * scale_factor
##                if channel_data[i] > 187500:
##                    channel_data[i] -= 2*187500
#            prog = int(sample[12] if kwargs['sample_rate']==250 else sample[11])
#            history.pop(0)
#            history.append(channel_data)
#            prog_history.pop(0)
#            prog_history.append(prog)
            
            channel_data = [0.0] * 8
            for i in channels:
                channel_data[i] = float(sample[i+1])
            trigger = int(sample[12] if kwargs['sample_rate']==250 else sample[11])
            history.pop(0)
            history.append(channel_data)
            trigger_history.pop(0)
            trigger_history.append(trigger)
            index_history.pop(0)
            index_history.append(index)
            
#            if trigger_history[-2] == 1.0 and trigger_history[-1] == 0.0:
#                recorded_count += 1
##                print index, channel_data

#        print 'Recorded:', recorded_count
#        if recorded_count >= 30 and False:
#            finished = True

#        data = np.array(history)
#        transformed = transform_fn(data, **kwargs)
#        if plot:
#            for i in range(8):
#                lines[i].set_data(history_x, transformed[:,i])
#            prog_line.set_data(history_x, np.array(prog_history) * bound * 2 - bound)
#
#        callback_fn(transformed if return_transformed else data)
#        
#        if finished: print 'Finished'
#
#        return lines + [prog_line] if plot else []
    
    
        data = np.array(history)

        transformed = transform_fn(data, **kwargs)
        if plot:
            max_std = max(np.std(transformed[-shown_size:], axis=0))
            ax.axis([-shown_size, 0, -2*max_std, 2*max_std])
            for i in channels:
                lines[i].set_data(history_x, transformed[-shown_size:,i])
            trigger_line.set_data(history_x, np.array(trigger_history[-shown_size:]) * 1.9*max_std)

        callback_fn(transformed if return_transformed else data,
                    np.array(trigger_history), np.array(index_history), sample_count, step)

        return list(lines[channels]) + [trigger_line] if plot else []

    line_ani = animation.FuncAnimation(fig, update, interval=0, blit=True)
    if plot: plt.show()

