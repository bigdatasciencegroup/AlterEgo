import numpy as np
import serial
import binascii
    
def start(device_name):
    with serial.Serial(device_name, 115200, timeout=1, parity=serial.PARITY_NONE,
                       stopbits=serial.STOPBITS_ONE) as ser:
        sent_commands = False
        while True:
            if ser.inWaiting():
                read = ser.read(ser.inWaiting())
                print read
                if '$$$' in read and not sent_commands:
                    sent_commands = True
                    print '\t', 'SENDING COMMANDS'
                    ser.write('A')

start('/dev/tty.usbserial-DQ007UBV')

