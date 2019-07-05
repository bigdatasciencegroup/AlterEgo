import numpy as np

import data

def on_data(history):
    print '\t'.join(map(str, history[-1]))

data.serial.start('/dev/tty.usbserial-DQ007UBV', on_data)