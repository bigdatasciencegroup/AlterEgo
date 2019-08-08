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


word_map = ["Turn on the lights", "Turn off the lights", "Change language", "Pause the music", "Resume", "Volume down", "Turn the lights on", "Switch on the lights", "Lights on", "Switch off the lights", "Turn the lights off", "Lights off", "Volume up", "Turn up the volume", "Turn the volume up", "Make the music louder", "Turn down the volume", "Turn the volume down", "Make the music softer", "Turn up the temperature", "Turn the temperature up", "Turn up the heat", "Turn the heat up", "Turn down the temperature", "Turn the temperature down", "Turn down the heat", "Turn the heat down", "Turn off the music", "Stop the music", "Play", "Put on the music", "Play the music", "Start the music", "Turn on the lamp", "Switch on the lamp", "Turn the lamp on", "Lamp on", "Turn off the lamp", "Turn the lamp off", "Switch off the lamp", "Lamp off", "Switch the lights on", "Turn on the kitchen lights", "Switch on the kitchen lights", "Turn the lights on in the kitchen", "Switch on the lights in the kitchen", "Lights on in the kitchen", "Kitchen lights on", "Switch the kitchen lights on", "Turn the kitchen lights on", "Turn on the lights in the bedroom", "Turn the bedroom lights on", "Switch on the lights in the bedroom", "Switch the bedroom lights on", "Lights on in the bedroom", "Bedroom lights on", "Turn on the washroom lights", "Turn the washroom lights on", "Lights on in the washroom", "Washroom lights on", "Turn on the bathroom lights", "Turn the bathroom lights on", "Switch on the bathroom lights", "Lights on in the bathroom", "Bathroom lights on", "Switch the lights off", "Turn off the kitchen lights", "Turn the lights off in the kitchen", "Switch off the lights in the kitchen", "Switch the lights off in the kitchen", "Lights off in the kitchen", "Kitchen lights off", "Turn off the lights in the bedroom", "Turn the bedroom lights off", "Switch off the lights in the bedroom", "Switch the bedroom lights off", "Lights off in the bedroom", "Bedroom lights off", "Turn off the washroom lights", "Switch off the washroom lights", "Lights off in the washroom", "Washroom lights off", "Turn volume up", "Turn sound up", "Turn it up", "Make it louder", "Louder", "Volume max", "Louder please", "Increase the volume", "Increase the sound volume", "I need volume", "Increase the sound", "Louder phone", "I can't hear that", "Too quiet", "That's too quiet", "Far too quiet", "Turn the sound up", "I need to hear this, increase the volume", "I couldn't hear anything, turn up the volume", "This video sound is too low, turn up the volume", "Decrease volume", "Turn volume down", "Turn sound down", "Decrease audio volume", "Reduce audio volume", "Turn it down", "Quieter", "Volume mute", "Lower the volume", "Volume lower", "Decrease the volume", "Decrease sound levels", "Too loud", "That's too loud", "Far too loud", "Make it quieter", "It's too loud, turn it down", "It's too loud, turn the volume down", "Increase the temperature", "Increase the heating", "Could you increase the heating?", "Could you increase the heating please?", "Make it hotter", "More heat", "Heat up", "Turn up the temperature in the kitchen", "Turn the kitchen temperature up", "Increase the temperature in the kitchen", "Increase the heating in the kitchen", "Turn up the heat in the kitchen", "Turn the heat up in the kitchen", "Kitchen heat up", "Turn up the temperature in the bedroom", "Turn the temperature in the bedroom up", "Turn up the bedroom heat", "Turn the bedroom heat up", "Bedroom heat up", "Increase the temperature in the bedroom", "Increase the heating in the bedroom", "Turn up the washroom temperature", "Turn the temperature up in the washroom", "Turn up the heat in the washroom", "Turn the heat up in the washroom", "Washroom heat up", "Increase the temperature in the washroom", "Increase the heating in the washroom", "Turn up the bathroom temperature", "Turn the temperature up in the bathroom", "Turn up the heat in the bathroom", "Turn the heat up in the bathroom", "Bathroom heat up", "Increase the temperature in the bathroom", "Increase the heating in the bathroom", "Make it cooler", "Less heat", "Heat down", "Decrease the temperature", "Decrease the heating", "Could you decrease the heating?", "Could you decrease the heating please?", "Turn down the temperature in the kitchen", "Turn the kitchen temperature down", "Turn down the heat in the kitchen", "Turn the heat down in the kitchen", "Kitchen heat down", "Decrease the temperature in the kitchen", "Decrease the heating in the kitchen", "Turn down the temperature in the bedroom", "Turn the temperature in the bedroom down", "Turn down the bedroom heat", "Turn the bedroom heat down", "Bedroom heat down", "Decrease the temperature in the bedroom", "Decrease the heating in the bedroom", "Turn down the washroom temperature", "Turn the temperature down in the washroom", "Turn down the heat in the washroom", "Turn the heat down in the washroom", "Decrease the temperature in the washroom", "Decrease the heating in the washroom", "Washroom heat down", "Turn down the bathroom temperature", "Turn the temperature down in the bathroom", "Turn down the heat in the bathroom", "Turn the heat down in the bathroom", "Decrease the temperature in the bathroom", "Decrease the heating in the bathroom", "Bathroom heat down", "Pause music", "Stop music", "Stop", "Play music", "Resume music", "Bring me the newspaper", "Get me the newspaper", "Bring the newspaper", "Bring newspaper", "Go get the newspaper", "Fetch the newspaper", "Bring me some juice", "Get me some juice", "Bring some juice", "Bring juice", "Go get me some juice", "Bring me my socks", "Get me my socks", "Bring socks", "Go get me my socks", "Fetch my socks", "Bring me my shoes", "Get me my shoes", "Bring shoes", "Bring my shoes", "Go get me my shoes", "Fetch my shoes", "Allow a different language", "Use a different language", "Change the language", "Switch the language", "Switch language", "Switch languages", "Change system language", "Open language settings", "Set the language", "Set language to Chinese", "Set language to German", "Set language to Korean", "Set language to English", "Set my device to Chinese", "OK now switch the main language to Chinese", "OK now switch the main language to German", "OK now switch the main language to Korean", "OK now switch the main language to English", "Set my phone's language to Chinese", "Set my phone's language to German", "Set my phone's language to Korean", "Set my phone's language to English", "I need to practice my Chinese. Switch the language", "I need to practice my German. Switch the language", "I need to practice my Korean. Switch the language", "I need to practice my English. Switch the language", "Turn the washroom lights off", "Pause", "Switch on the washroom lights", "Bring my socks", "Language settings", "Turn lights on", "Turn light on", "Turn light off", "Turn lights off", "Switch lights on", "Switch light on", "Switch light off", "Switch lights off", "Decrease temperature", "Increase temperature", "Decrease heating", "Increase heating", "Fetch newspaper", "Get newspaper", "Get shoes", "Can't hear", "Cannot hear", "Mute sound", "Mute volume", "Kitchen temperature down", "Kitchen temperature up", "Bedroom temperature down", "Bedroom temperature up", "Switch language to Chinese", "Switch language to German", "Switch language to Korean", "Switch language to English", "Thank you", "Get up", "Help me get up", "Please help me get up", "Help me get up please", "Help me up", "Help me up please", "Please help me up", "Hello", "Hi", "Hey", "Good morning", "Good evening", "Good afternoon", "Goodbye", "Bye", "See you", "See you soon", "See you later", "Help", "Help me", "Help me please", "Please help me", "Can you help me?", "Can you please help me?", "Please, can you help me?", "Could you help me?", "Could you please help me?", "Bathroom", "I need to go to the bathroom", "Please help me go to the bathroom", "Help me go to the bathroom", "I'm tired", "I'm so tired", "I'm too tired", "I need to get some sleep", "I need a nap", "I need to take a nap", "I want to take a nap", "I need to sleep", "I want to sleep", "I need to lie down", "Please help me lie down", "Help me lie down", "Please help me get into bed", "Help me get into bed", "I have a headache", "My head hurts", "I'm hungry", "I want to eat", "I need to eat", "I want to eat something", "I need to eat something", "I'm thirsty", "I need to drink", "I want to drink something", "I need to drink something", "Get me water", "Fetch me water", "Bring me water", "Bring me water please", "Please get me water", "Get me some water", "Get me some water please", "I have a backache", "My back aches", "My back hurts", "I want to watch TV", "Turn on the TV", "Turn the TV on", "Turn TV on", "TV on", "Please turn on the TV", "Turn on the TV, please", "It hurts", "I'm in pain", "I feel pain in my back", "Turn on the computer", "Turn computer on", "Turn the computer on", "Turn computer off", "Switch computer off", "Turn the computer off", "Restart computer", "Reboot computer", "Restart my computer", "Reboot my computer", "Restart the computer", "Reboot the computer", "Turn on computer", "Turn off computer", "I can't hear it", "Can't hear anything", "Cannot hear anything", "I cannot hear it", "Bathroom temperature down", "Bathroom temperature up", "Please switch off the lamp", "Please switch on the lamp", "Thank you so much", "Thanks", "Thank you very much", "I am in pain", "It hurts a lot", "It hurts so much", "I am in a lot of pain", "I'm in a lot of pain", "I'm exhausted", "I'm so exhausted", "I am tired", "I am so tired", "I am too tired", "I am exhausted", "I am so exhausted", "I am thirsty", "I'm very thirsty", "I'm having a headache", "I'm getting a headache", "I have a bad beadache", "My back is hurting", "I have back pains", "My back hurts so much", "I have to go to the bathroom", "Please turn the computer off", "Please turn the computer on",  'FINISHED FINISHED FINISHED FINISHED FINISHED']

# Change labels as per the words to record - copy paste from label_generator_for_record_serial.py
#labels = generate_labels(len(word_map))

"""
Recording Session Utkarsh 1:
[360, 315, 105, 233, 287, 186, 265, 337, 293, 153, 335, 323, 51, 109, 259, 93, 94, 82, 388, 296, 126, 284, 392, 170, 189, 347, 17, 107, 64, 44, 157, 187, 282, 384, 195, 294, 211, 79, 45, 280]
[326, 27, 273, 386, 143, 210, 382, 48, 355, 343, 288, 367, 131, 368, 169, 115, 24, 290, 313, 141, 30, 321, 194, 12, 182, 101, 341, 121, 320, 251, 205, 399, 158, 155, 127, 264, 311, 365, 254, 372]
[39, 22, 1, 240, 162, 237, 103, 262, 146, 96, 154, 78, 309, 252, 324, 228, 49, 136, 137, 34, 333, 397, 242, 279, 393, 181, 3, 234, 133, 310, 319, 215, 191, 356, 230, 248, 176, 350, 11, 381]
[140, 139, 197, 289, 297, 29, 340, 6, 218, 338, 261, 325, 241, 292, 161, 266, 256, 59, 72, 69, 52, 330, 349, 387, 167, 35, 329, 92, 244, 217, 80, 209, 62, 31, 312, 391, 291, 255, 247, 60]
[168, 2, 236, 150, 56, 223, 37, 208, 300, 257, 226, 304, 327, 200, 175, 204, 47, 118, 142, 348, 113, 108, 270, 43, 148, 132, 299, 61, 36, 213, 198, 219, 165, 246, 102, 77, 38, 267, 258, 201]
[183, 110, 88, 269, 46, 263, 278, 68, 144, 390, 128, 171, 174, 354, 122, 253, 389, 331, 54, 151, 91, 67, 363, 352, 138, 97, 70, 277, 396, 220, 318, 357, 75, 178, 308, 123, 231, 301, 369, 199]
[83, 272, 58, 314, 166, 276, 5, 7, 86, 124, 239, 268, 378, 42, 374, 305, 229, 20, 87, 179, 364, 173, 71, 362, 281, 207, 316, 117, 28, 159, 23, 317, 380, 245, 302, 0, 116, 84, 202, 85]
[232, 145, 119, 344, 359, 98, 214, 328, 249, 298, 13, 395, 57, 134, 152, 375, 66, 185, 149, 398, 196, 283, 346, 361, 274, 26, 303, 4, 250, 106, 156, 95, 135, 206, 8, 394, 224, 358, 222, 307]
[32, 129, 286, 90, 9, 193, 271, 120, 379, 260, 14, 111, 81, 334, 41, 180, 55, 353, 339, 190, 227, 112, 275, 212, 125, 216, 332, 376, 114, 385, 130, 65, 184, 15, 351, 383, 53, 225, 235, 366]
[322, 33, 306, 100, 243, 285, 50, 188, 104, 342, 336, 19, 73, 373, 160, 99, 74, 16, 238, 221, 370, 163, 345, 203, 172, 177, 371, 25, 40, 10, 147, 295, 18, 377, 76, 164, 21, 89, 63, 192]
----------------------------------------
Recording Session Utkarsh 2:
[287, 189, 62, 387, 74, 193, 354, 273, 5, 170, 286, 248, 328, 288, 46, 337, 285, 249, 343, 4, 18, 204, 94, 124, 281, 279, 318, 101, 120, 219, 321, 197, 301, 73, 376, 66, 283, 7, 157, 25]
[195, 106, 230, 23, 379, 109, 309, 132, 135, 133, 19, 310, 367, 169, 103, 390, 79, 266, 385, 306, 203, 26, 294, 312, 29, 8, 30, 171, 325, 183, 320, 1, 126, 259, 49, 161, 146, 307, 131, 47]
[198, 391, 224, 97, 234, 92, 56, 262, 335, 122, 211, 16, 98, 188, 300, 40, 260, 164, 342, 221, 289, 302, 149, 76, 2, 257, 207, 15, 156, 356, 363, 95, 186, 341, 48, 338, 39, 368, 292, 381]
[276, 393, 70, 33, 269, 53, 162, 145, 148, 43, 290, 275, 209, 129, 54, 322, 179, 190, 154, 44, 176, 99, 159, 181, 305, 31, 280, 115, 118, 308, 178, 184, 347, 28, 258, 111, 231, 143, 163, 104]
[63, 378, 168, 226, 265, 38, 139, 291, 251, 114, 52, 140, 220, 252, 20, 182, 313, 174, 21, 395, 134, 227, 215, 245, 263, 85, 205, 240, 199, 332, 128, 217, 201, 303, 278, 187, 233, 192, 277, 299]
[151, 382, 222, 173, 241, 238, 352, 130, 22, 107, 121, 360, 45, 386, 32, 264, 304, 35, 348, 316, 87, 365, 228, 295, 270, 9, 77, 13, 175, 160, 117, 196, 353, 212, 24, 65, 172, 261, 323, 177]
[125, 327, 213, 136, 90, 216, 72, 105, 229, 374, 41, 218, 317, 350, 185, 64, 331, 297, 225, 397, 369, 80, 399, 42, 357, 326, 346, 142, 17, 116, 355, 123, 58, 330, 246, 68, 71, 244, 366, 137]
[113, 315, 371, 34, 55, 314, 67, 237, 336, 268, 10, 236, 12, 3, 37, 373, 57, 340, 61, 36, 194, 255, 247, 298, 59, 75, 362, 147, 282, 389, 333, 82, 202, 254, 345, 86, 210, 180, 344, 398]
[375, 388, 370, 150, 396, 93, 88, 152, 11, 69, 108, 242, 235, 377, 256, 267, 102, 253, 284, 91, 311, 329, 223, 349, 380, 358, 293, 127, 250, 51, 96, 153, 119, 141, 0, 384, 100, 191, 214, 324]
[200, 27, 364, 361, 239, 359, 50, 383, 271, 89, 166, 272, 167, 339, 351, 110, 78, 232, 14, 83, 319, 394, 81, 138, 372, 155, 392, 334, 206, 60, 165, 243, 112, 158, 208, 144, 6, 84, 274, 296]
"""

labels = [200, 27, 364, 361, 239, 359, 50, 383, 271, 89, 166, 272, 167, 339, 351, 110, 78, 232, 14, 83, 319, 394, 81, 138, 372, 155, 392, 334, 206, 60, 165, 243, 112, 158, 208, 144, 6, 84, 274, 296]

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
