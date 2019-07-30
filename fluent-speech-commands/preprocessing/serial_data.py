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
import datetime

import transform

ADS1299_Vref = 4.5  #reference voltage for ADC in ADS1299.  set by its hardware
ADS1299_gain = 24.0  #assumed gain setting for ADS1299.  set by its Arduino code
scale_factor = ADS1299_Vref/float((pow(2,23)-1))/ADS1299_gain*1000000.

packet_size = 31

history_size = 1800
shown_size = 600

buffer_sizes = [0.0] * 100

count = 0
step = 50

sample_count = 0
recorded_count = 0

def start(device_name, callback_fn, channels=range(0, 8), bipolar=False, history_size=history_size, shown_size=shown_size,
          transform_fn=transform.default_transform, plot=True, return_transformed=True,
          override_step=None, **kwargs):
    history = [[0.0] * 8 for i in range(history_size)]
    trigger_history = [0.0] * history_size
    index_history = [0] * history_size
    global step
    if override_step:
        step = override_step

    fig = plt.figure(0, figsize=(12, 8))
    if plot:
        colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
        ax = fig.gca()
        trigger_line = ax.plot([-shown_size, 0], [0, 0], '--', lw=1.0, c='black', alpha=0.5)[0]
        lines = np.array(
            [(ax.plot([],[], '-', lw=1.5, c=colors[i])[0] if i in channels else None) for i in range(8)])
        #line = ax.plot([],[], '-', lw=0.5, c=colors[1])[0]
        ax.set_title('Serial Data')
        ax.set_xlabel('Sample')
#        ax.axis([-shown_size, 0, -187500/8/128, 187500/8/128])
#        if kwargs.get('apply_subtract_mean', transform.apply_subtract_mean) \
#            or kwargs.get('apply_bandpass_filter', transform.apply_bandpass_filter):
#            ax.axis([-shown_size, 0, -187500/8/128, 187500/8/128])
#        else:
##            ax.axis([-history_size, 0, 0, 187500*2])
#            ax.axis([-shown_size, 0, -187500, 187500])


        ax.axis([-shown_size, 0, -1.0 * len(channels), 1])

        infos = [ax.text(0.005, 0.975 - 0.03*i, ' Channel ' + str(i+1) + ' ',
                         color=('black' if i in channels else 'white'),
                         bbox={'facecolor': (colors[i] if i in channels else 'black'),
                               'alpha':0.5 if i in channels else 0.2, 'pad':1},
                         transform=ax.transAxes, ha='left')\
                 for i in range(8)]

        ax.get_yaxis().set_ticks([])

        plt.subplots_adjust(left=0.04, right=0.96, bottom=0.09, top=0.92)

        history_x = range(-shown_size, 0)

    with serial.Serial(device_name, 115200, timeout=1, parity=serial.PARITY_NONE,
                       stopbits=serial.STOPBITS_ONE) as ser:
#        ser.write('1')
        for i in range(8):
            if i not in channels:
                ser.write(str(i+1))
        ser.write('/3')
        if bipolar:
            for i in range(8):
                if i not in channels:
                    ser.write('x' + str(i+1) + '060100X')
        ser.write('b')

        filename = 'serial_data/data_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt'
        file_object = open(filename, 'a')

        file_object.write('%OpenBCI Raw EEG Data From Serial Read\n')
        file_object.write('%Number of channels = 8\n')
        file_object.write('%Sample Rate = 250.0 Hz\n')
        file_object.write('%First Column = SampleIndex\n')
        file_object.write('%Last Column = Timestamp\n')
        file_object.write('%Other Columns = EEG data in microvolts followed by digital read data\n')

        def update(i):
            global count
            global sample_count
            global buffer_sizes
            global step
            global recorded_count

            start_count = count
            buffer_sizes.pop(0)
            buffer_sizes.append(ser.inWaiting())
            mean_buffer_size = np.mean(buffer_sizes)
#            print mean_buffer_size

            if not override_step:
                if mean_buffer_size > 950: # 900
                    step += 1
                    buffer_sizes = [880.0] * len(buffer_sizes)
                if mean_buffer_size < 879: # 860
                    step = max(1, step - 1)
                    buffer_sizes = [880.0] * len(buffer_sizes)
#                print step

            while ser.inWaiting() and count - start_count < step:
                count += 1
                s = ser.read(1)
                if binascii.hexlify(s) == 'a0':
                    sample_count += 1
                    sample = ser.read(packet_size)
#                    index = int(binascii.hexlify(sample[0:1]), 16)
                    index = sample_count
                    channel_data = [0.0] * 8
                    for i in channels:
                        channel_data[i] = int(binascii.hexlify(sample[3*i+1:3*i+4]), 16)*scale_factor
                        if channel_data[i] > 187500:
                            channel_data[i] -= 2*187500
                    trigger = int(bool(int(binascii.hexlify(sample[28:29]), 16)))
                    file_object.write(', '.join(map(str, [sample_count] + channel_data + [0, 1, 1, trigger, 1,
                                                            datetime.datetime.now().strftime("%H:%M:%S.%f")])) + '\n')
                    history.pop(0)
                    history.append(channel_data)
                    trigger_history.pop(0)
                    trigger_history.append(trigger)
                    index_history.pop(0)
                    index_history.append(index)
    #                print index, channel_data

                    if trigger_history[-2] == 1.0 and trigger_history[-1] == 0.0 \
                        and np.all(np.array(trigger_history[-50:-1]) == 1.0) and sample_count > 20:
                        recorded_count += 1

            data = np.array(history)

            transformed = transform_fn(data, **kwargs)
            if plot:
                max_vals = [5 * np.std(transformed[-shown_size:,i], axis=0) for i in range(8)]
#                max_val = 1.5 * np.max(np.abs(transformed[-shown_size:]))
                count = -1
                for i in channels:
                    count += 1
                    lines[i].set_data(history_x, transformed[-shown_size:,i] / max_vals[i] - 1.0 * count)
                trigger_line.set_data(history_x,
                                      np.array(trigger_history[-shown_size:]) * (len(channels)+3) - len(channels)-1)

            callback_fn(transformed if return_transformed else data,
                        np.array(trigger_history), np.array(index_history), sample_count, step, recorded_count)

            return list(lines[channels]) + [trigger_line] if plot else []

        line_ani = animation.FuncAnimation(fig, update, interval=0, blit=True)
        if plot: plt.show()
