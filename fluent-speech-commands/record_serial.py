'''
Ran using Python 2.7.16

Records Data from OpenBCI Board (switched to PC) over Bluetooth using the OpenBCI Dongle at 250Hz
The data plotted on the GUI in realtime is after preprocessing/transformation
The data saved in the datetime file in serial_data folder is without the preprocessing/transformation
Make sure you have a folder 'serial_data' in the same location as this script
'''

import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import os

import preprocessing
from label_generator_for_record_serial import generate_labels

channels = range(0, 8)  # Must be same as trained model if test_model==True


def transform_data(sequence_groups, sample_rate=250):
    #### Apply DC offset and drift correction
    drift_low_freq = 0.5  # 0.5
    sequence_groups = preprocessing.transform.subtract_initial(sequence_groups)
    sequence_groups = preprocessing.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    sequence_groups = preprocessing.transform.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq
    notch_freq = 60
    num_times = 3  # pretty much just the filter order
    freqs = map(int, map(round, np.arange(1, sample_rate / (2. * notch_freq)) * notch_freq))
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = preprocessing.transform.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    # sequence_groups = data.transform.normalize_std(sequence_groups)

    def normalize_kernel(kernel, subtract_mean=False):
        if subtract_mean:
            kernel = np.array(kernel, np.float32) - np.mean(kernel)
        return np.array(kernel, np.float32) / np.sum(np.abs(kernel))

    def ricker_function(t, sigma):
        return 2. / (np.sqrt(3 * sigma) * np.pi ** 0.25) * (1. - (float(t) / sigma) ** 2) * np.exp(
            -(float(t) ** 2) / (2 * sigma ** 2))

    def ricker_wavelet(n, sigma):
        return np.array(map(lambda x: ricker_function(x, sigma), range(-n // 2, n // 2 + 1)))

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = preprocessing.transform.correlate(sequence_groups, ricker_kernel)
    ricker_subtraction_multiplier = 2.0
    sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

    #### Apply sine wavelet kernel
    #    period = int(sample_rate)
    #    sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
    #    sequence_groups = data.transform.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5
    high_freq = 8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = preprocessing.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

    #### Apply hard bandpassing
    #    sequence_groups = data.transform.fft(sequence_groups)
    #    sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
    #    sequence_groups = np.real(data.transform.ifft(sequence_groups))

    return sequence_groups


word_map = ["Turn on the lights", "Turn off the lights", "Change language", "Pause the music", "Resume", "Volume down", "Turn the lights on", "Switch on the lights", "Lights on", "Switch off the lights", "Turn the lights off", "Lights off", "Volume up", "Turn up the volume", "Turn the volume up", "Make the music louder", "Turn down the volume", "Turn the volume down", "Make the music softer", "Turn up the temperature", "Turn the temperature up", "Turn up the heat", "Turn the heat up", "Turn down the temperature", "Turn the temperature down", "Turn down the heat", "Turn the heat down", "Turn off the music", "Stop the music", "Play", "Put on the music", "Play the music", "Start the music", "Turn on the lamp", "Switch on the lamp", "Turn the lamp on", "Lamp on", "Turn off the lamp", "Turn the lamp off", "Switch off the lamp", "Lamp off", "Switch the lights on", "Turn on the kitchen lights", "Switch on the kitchen lights", "Turn the lights on in the kitchen", "Switch on the lights in the kitchen", "Lights on in the kitchen", "Kitchen lights on", "Switch the kitchen lights on", "Turn the kitchen lights on", "Turn on the lights in the bedroom", "Turn the bedroom lights on", "Switch on the lights in the bedroom", "Switch the bedroom lights on", "Lights on in the bedroom", "Bedroom lights on", "Turn on the washroom lights", "Turn the washroom lights on", "Lights on in the washroom", "Washroom lights on", "Turn on the bathroom lights", "Turn the bathroom lights on", "Switch on the bathroom lights", "Lights on in the bathroom", "Bathroom lights on", "Switch the lights off", "Turn off the kitchen lights", "Turn the lights off in the kitchen", "Switch off the lights in the kitchen", "Switch the lights off in the kitchen", "Lights off in the kitchen", "Kitchen lights off", "Turn off the lights in the bedroom", "Turn the bedroom lights off", "Switch off the lights in the bedroom", "Switch the bedroom lights off", "Lights off in the bedroom", "Bedroom lights off", "Turn off the washroom lights", "Switch off the washroom lights", "Lights off in the washroom", "Washroom lights off", "Turn volume up", "Turn sound up", "Turn it up", "Make it louder", "Louder", "Volume max", "Louder please", "Increase the volume", "Increase the sound volume", "I need volume", "Increase the sound", "Louder phone", "I can't hear that", "Too quiet", "That's too quiet", "Far too quiet", "Turn the sound up", "I need to hear this, increase the volume", "I couldn't hear anything, turn up the volume", "This video sound is too low, turn up the volume", "Decrease volume", "Turn volume down", "Turn sound down", "Decrease audio volume", "Reduce audio volume", "Turn it down", "Quieter", "Volume mute", "Lower the volume", "Volume lower", "Decrease the volume", "Decrease sound levels", "Too loud", "That's too loud", "Far too loud", "Make it quieter", "It's too loud, turn it down", "It's too loud, turn the volume down", "Increase the temperature", "Increase the heating", "Could you increase the heating?", "Could you increase the heating please?", "Make it hotter", "More heat", "Heat up", "Turn up the temperature in the kitchen", "Turn the kitchen temperature up", "Increase the temperature in the kitchen", "Increase the heating in the kitchen", "Turn up the heat in the kitchen", "Turn the heat up in the kitchen", "Kitchen heat up", "Turn up the temperature in the bedroom", "Turn the temperature in the bedroom up", "Turn up the bedroom heat", "Turn the bedroom heat up", "Bedroom heat up", "Increase the temperature in the bedroom", "Increase the heating in the bedroom", "Turn up the washroom temperature", "Turn the temperature up in the washroom", "Turn up the heat in the washroom", "Turn the heat up in the washroom", "Washroom heat up", "Increase the temperature in the washroom", "Increase the heating in the washroom", "Turn up the bathroom temperature", "Turn the temperature up in the bathroom", "Turn up the heat in the bathroom", "Turn the heat up in the bathroom", "Bathroom heat up", "Increase the temperature in the bathroom", "Increase the heating in the bathroom", "Make it cooler", "Less heat", "Heat down", "Decrease the temperature", "Decrease the heating", "Could you decrease the heating?", "Could you decrease the heating please?", "Turn down the temperature in the kitchen", "Turn the kitchen temperature down", "Turn down the heat in the kitchen", "Turn the heat down in the kitchen", "Kitchen heat down", "Decrease the temperature in the kitchen", "Decrease the heating in the kitchen", "Turn down the temperature in the bedroom", "Turn the temperature in the bedroom down", "Turn down the bedroom heat", "Turn the bedroom heat down", "Bedroom heat down", "Decrease the temperature in the bedroom", "Decrease the heating in the bedroom", "Turn down the washroom temperature", "Turn the temperature down in the washroom", "Turn down the heat in the washroom", "Turn the heat down in the washroom", "Decrease the temperature in the washroom", "Decrease the heating in the washroom", "Washroom heat down", "Turn down the bathroom temperature", "Turn the temperature down in the bathroom", "Turn down the heat in the bathroom", "Turn the heat down in the bathroom", "Decrease the temperature in the bathroom", "Decrease the heating in the bathroom", "Bathroom heat down", "Pause music", "Stop music", "Stop", "Play music", "Resume music", "Bring me the newspaper", "Get me the newspaper", "Bring the newspaper", "Bring newspaper", "Go get the newspaper", "Fetch the newspaper", "Bring me some juice", "Get me some juice", "Bring some juice", "Bring juice", "Go get me some juice", "Bring me my socks", "Get me my socks", "Bring socks", "Go get me my socks", "Fetch my socks", "Bring me my shoes", "Get me my shoes", "Bring shoes", "Bring my shoes", "Go get me my shoes", "Fetch my shoes", "Allow a different language", "Use a different language", "Change the language", "Switch the language", "Switch language", "Switch languages", "Change system language", "Open language settings", "Set the language", "Set language to Chinese", "Set language to German", "Set language to Korean", "Set language to English", "Set my device to Chinese", "OK now switch the main language to Chinese", "OK now switch the main language to German", "OK now switch the main language to Korean", "OK now switch the main language to English", "Set my phone's language to Chinese", "Set my phone's language to German", "Set my phone's language to Korean", "Set my phone's language to English", "I need to practice my Chinese. Switch the language", "I need to practice my German. Switch the language", "I need to practice my Korean. Switch the language", "I need to practice my English. Switch the language", "Turn the washroom lights off", "Pause", "Switch on the washroom lights", "Bring my socks", "Language settings", "Turn lights on", "Turn light on", "Turn light off", "Turn lights off", "Switch lights on", "Switch light on", "Switch light off", "Switch lights off", "Decrease temperature", "Increase temperature", "Decrease heating", "Increase heating", "Fetch newspaper", "Get newspaper", "Get shoes", "Can't hear", "Cannot hear", "Mute sound", "Mute volume", "Kitchen temperature down", "Kitchen temperature up", "Bedroom temperature down", "Bedroom temperature up", "Switch language to Chinese", "Switch language to German", "Switch language to Korean", "Switch language to English", "Thank you", "Get up", "Help me get up", "Please help me get up", "Help me get up please", "Help me up", "Help me up please", "Please help me up", "Hello", "Hi", "Hey", "Good morning", "Good evening", "Good afternoon", "Goodbye", "Bye", "See you", "See you soon", "See you later", "Help", "Help me", "Help me please", "Please help me", "Can you help me?", "Can you please help me?", "Please, can you help me?", "Could you help me?", "Could you please help me?", "Bathroom", "I need to go to the bathroom", "Please help me go to the bathroom", "Help me go to the bathroom", "I'm tired", "I'm so tired", "I'm too tired", "I need to get some sleep", "I need a nap", "I need to take a nap", "I want to take a nap", "I need to sleep", "I want to sleep", "I need to lie down", "Please help me lie down", "Help me lie down", "Please help me get into bed", "Help me get into bed", "I have a headache", "My head hurts", "I'm hungry", "I want to eat", "I need to eat", "I want to eat something", "I need to eat something", "I'm thirsty", "I need to drink", "I want to drink something", "I need to drink something", "Get me water", "Fetch me water", "Bring me water", "Bring me water please", "Please get me water", "Get me some water", "Get me some water please", "I have a backache", "My back aches", "My back hurts", "I want to watch TV", "Turn on the TV", "Turn the TV on", "Turn TV on", "TV on", "Please turn on the TV", "Turn on the TV, please", "It hurts", "I'm in pain", "I feel pain in my back", "Turn on the computer", "Turn computer on", "Turn the computer on", "Turn computer off", "Switch computer off", "Turn the computer off", "Restart computer", "Reboot computer", "Restart my computer", "Reboot my computer", "Restart the computer", "Reboot the computer", "Turn on computer", "Turn off computer", 'FINISHED FINISHED FINISHED FINISHED FINISHED']

# Change labels as per the words to record - copy paste from label_generator_for_record_serial.py
#labels = generate_labels(len(word_map))

"""
Recordings Utkarsh 1:
[197, 241, 16, 61, 338, 69, 337, 227, 302, 295, 265, 14, 244, 99, 263, 247, 134, 275, 271, 102, 357, 54, 238, 214, 351, 331, 329, 278, 85, 184, 46, 62, 347, 41, 187, 108]
[126, 88, 269, 326, 131, 307, 51, 167, 303, 237, 87, 318, 204, 188, 150, 353, 292, 296, 65, 355, 83, 122, 78, 254, 168, 359, 274, 164, 45, 141, 313, 97, 49, 10, 287, 47, 43]
[86, 299, 286, 297, 175, 90, 59, 109, 125, 114, 259, 82, 281, 180, 21, 25, 149, 266, 143, 328, 320, 255, 199, 319, 115, 288, 256, 156, 165, 29, 44, 354, 235, 224, 91, 147]
[166, 298, 333, 11, 160, 137, 243, 315, 232, 308, 157, 212, 207, 121, 74, 63, 321, 163, 192, 96, 362, 215, 230, 19, 363, 130, 118, 344, 336, 136, 272, 262, 285, 200, 251, 64, 66]
[170, 158, 348, 129, 332, 293, 261, 176, 120, 258, 189, 6, 233, 24, 101, 209, 306, 35, 208, 75, 124, 103, 93, 283, 342, 26, 117, 30, 181, 205, 23, 92, 111, 70, 50, 226, 155]
[276, 133, 110, 104, 20, 195, 169, 325, 268, 98, 270, 95, 210, 305, 231, 178, 67, 154, 28, 52, 317, 250, 4, 245, 316, 22, 100, 191, 116, 17, 301, 139, 172, 314, 77, 327]
[312, 106, 40, 138, 119, 123, 148, 190, 80, 56, 330, 72, 219, 12, 42, 352, 218, 264, 267, 220, 2, 13, 84, 94, 60, 68, 201, 339, 198, 345, 18, 202, 213, 228, 140, 194, 145]
[248, 76, 146, 260, 229, 242, 193, 183, 1, 304, 346, 113, 310, 7, 234, 203, 159, 38, 39, 324, 335, 311, 309, 71, 291, 151, 132, 257, 252, 33, 253, 171, 211, 48, 142, 0]
[112, 279, 364, 343, 221, 222, 58, 322, 290, 186, 356, 177, 174, 135, 300, 282, 239, 323, 280, 349, 206, 246, 127, 9, 152, 31, 273, 107, 27, 162, 173, 36, 225, 358, 182, 3, 153]
[361, 365, 289, 105, 284, 216, 37, 5, 81, 341, 15, 161, 89, 350, 34, 294, 79, 223, 179, 340, 185, 240, 53, 196, 217, 32, 236, 55, 334, 144, 128, 277, 57, 73, 360, 249, 8]
-------------------------------------------------------------------------------------------------------------------------
Recordings Utkarsh 2:
[197, 241, 16, 61, 338, 69, 337, 227, 302, 295, 265, 14, 244, 99, 263, 247, 134, 275, 271, 102, 357, 54, 238, 214, 351, 331, 329, 278, 85, 184, 46, 62, 347, 41, 187, 108]
[126, 88, 269, 326, 131, 307, 51, 167, 303, 237, 87, 318, 204, 188, 150, 353, 292, 296, 65, 355, 83, 122, 78, 254, 168, 359, 274, 164, 45, 141, 313, 97, 49, 10, 287, 47, 43]
[86, 299, 286, 297, 175, 90, 59, 109, 125, 114, 259, 82, 281, 180, 21, 25, 149, 266, 143, 328, 320, 255, 199, 319, 115, 288, 256, 156, 165, 29, 44, 354, 235, 224, 91, 147]
[166, 298, 333, 11, 160, 137, 243, 315, 232, 308, 157, 212, 207, 121, 74, 63, 321, 163, 192, 96, 362, 215, 230, 19, 363, 130, 118, 344, 336, 136, 272, 262, 285, 200, 251, 64, 66]
[170, 158, 348, 129, 332, 293, 261, 176, 120, 258, 189, 6, 233, 24, 101, 209, 306, 35, 208, 75, 124, 103, 93, 283, 342, 26, 117, 30, 181, 205, 23, 92, 111, 70, 50, 226, 155]
[276, 133, 110, 104, 20, 195, 169, 325, 268, 98, 270, 95, 210, 305, 231, 178, 67, 154, 28, 52, 317, 250, 4, 245, 316, 22, 100, 191, 116, 17, 301, 139, 172, 314, 77, 327]
[312, 106, 40, 138, 119, 123, 148, 190, 80, 56, 330, 72, 219, 12, 42, 352, 218, 264, 267, 220, 2, 13, 84, 94, 60, 68, 201, 339, 198, 345, 18, 202, 213, 228, 140, 194, 145]
[248, 76, 146, 260, 229, 242, 193, 183, 1, 304, 346, 113, 310, 7, 234, 203, 159, 38, 39, 324, 335, 311, 309, 71, 291, 151, 132, 257, 252, 33, 253, 171, 211, 48, 142, 0]
[112, 279, 364, 343, 221, 222, 58, 322, 290, 186, 356, 177, 174, 135, 300, 282, 239, 323, 280, 349, 206, 246, 127, 9, 152, 31, 273, 107, 27, 162, 173, 36, 225, 358, 182, 3, 153]
[361, 365, 289, 105, 284, 216, 37, 5, 81, 341, 15, 161, 89, 350, 34, 294, 79, 223, 179, 340, 185, 240, 53, 196, 217, 32, 236, 55, 334, 144, 128, 277, 57, 73, 360, 249, 8]
"""

labels = [273, 213, 14, 12, 121, 67, 94, 96, 56, 344, 117, 58, 287, 230, 138, 328, 109, 151, 71, 147, 285, 252, 347, 202, 199, 115, 77, 227, 150, 98, 45, 359, 317, 63, 348, 201, 126]

labels = labels + [-1]

recorded_length = None
last_recorded_count = -1


def on_data(history, trigger_history, index_history, count, samples_per_update, recorded_count):
    global last_recorded_count
    global recorded_length
    if recorded_count > last_recorded_count:
        os.system('say "' + word_map[labels[recorded_count]] + '" &')
    last_recorded_count = recorded_count
    print
    print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i + 1) for i in range(8)])
    print str('{:.1f}'.format(count / 250.)) + 's\t\t' + '\t'.join(
        map(lambda (i, x): '{:f}'.format(x) if i in channels else '--\t', enumerate(history[-1])))
    print
    if recorded_count > 0:
        start, end = None, None
        for i in range(len(trigger_history))[::-1]:
            if trigger_history[i] and end is None:
                end = i
            elif not trigger_history[i] and end:
                start = i
                break
        if start and end:
            recorded_length = end - start

            per_word_rec_length = len(word_map[labels[recorded_count]].split()) / recorded_length
            print recorded_length
            print len(word_map[labels[recorded_count]].split())
            #recorded_length = len(word_map[labels[recorded_count]].split())
        print 'WPM:', 60.0 / (float(recorded_length) / 250 / len(word_map[labels[recorded_count-(1 if end < len(trigger_history)-1 else 0)]].split(' ')))

        #print 'WPM:', 60.0 / (float(recorded_length) / 250)
    print
    print 'Sample #' + str(recorded_count + 1) + '/' + str(len(labels) - 1), '\tNext:', word_map[labels[recorded_count]]
    print


preprocessing.serial.start('/dev/tty.usbserial-DM01HUN9',
                  on_data, channels=channels, transform_fn=transform_data,
                  history_size=2500, shown_size=1200, override_step=100, bipolar=False)  # 35
