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

fs = 250

step = 25 #10 #25
#step = 15 #10 #25
count = 0
finished = False
recorded_count = 0
    
def start(filepath, callback_fn, plot=True, return_transformed=True, speed=1.0,
          transform_fn=transform.default_transform, default_step=step, starting_point=0, **kwargs):
    
    history_size = 1000 * kwargs.get('sample_rate', 250) // 250 #1000
    history = [[0.0] * 8 for i in range(history_size)]
    prog_history = [0] * history_size
    
    global step
    step = default_step
    contents = map(lambda x: x.strip(), open(filepath, 'r').readlines())
    frames = filter(lambda x: x and x[0] != '%', contents)
    frames = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames)[starting_point:]

    fig = plt.figure(0)
    if plot:
        colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
        ax = fig.gca()
        lines = [ax.plot([],[], '-', lw=0.5, c=colors[i])[0] for i in range(8)]
        prog_line = ax.plot([], [], 'k', lw=1.0)[0]
        #line = ax.plot([],[], '-', lw=0.5, c=colors[1])[0]
        ax.set_title('Channel Data')
        ax.set_xlabel('Sample')
        bound = 187500
        if kwargs.get('apply_subtract_mean', transform.apply_subtract_mean) \
            or kwargs.get('apply_bandpass_filter', transform.apply_bandpass_filter):
            bound = 187500/8/128
        ax.axis([-history_size, 0, -bound, bound])

        history_x = range(-history_size, 0)

    def update(i):
        global count
        global elapseds
        global timestamp
        global finished
        global recorded_count
        
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
            sample = frames[count * step + l]
            index = int(sample[0])
            channel_data = [0.0] * 8
            for i in range(8):
                channel_data[i] = float(sample[i+1]) # * scale_factor
#                if channel_data[i] > 187500:
#                    channel_data[i] -= 2*187500
            prog = int(sample[12] if kwargs['sample_rate']==250 else sample[11])
            history.pop(0)
            history.append(channel_data)
            prog_history.pop(0)
            prog_history.append(prog)
            
            if prog_history[-2] == 1.0 and prog_history[-1] == 0.0:
                recorded_count += 1
#                print index, channel_data

        print 'Recorded:', recorded_count
        if recorded_count >= 30 and False:
            finished = True

        data = np.array(history)
        transformed = transform_fn(data, **kwargs)
        if plot:
            for i in range(8):
                lines[i].set_data(history_x, transformed[:,i])
            prog_line.set_data(history_x, np.array(prog_history) * bound * 2 - bound)

        callback_fn(transformed if return_transformed else data)
        
        if finished: print 'Finished'

        return lines + [prog_line] if plot else []

    line_ani = animation.FuncAnimation(fig, update, interval=0, blit=True)
    if plot: plt.show()

