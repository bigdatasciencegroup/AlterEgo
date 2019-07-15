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


word_map = ["Turn on the lights,Turn,on,the,lights", "Turn off the lights,Turn,off,the,lights", "Change language,Change,language", "Pause the music,Pause,the,music", "Resume,Resume", "Volume down,Volume,down", "Turn the lights on,Turn,the,lights,on", "Switch on the lights,Switch,on,the,lights", "Lights on,Lights,on", "Switch off the lights,Switch,off,the,lights", "Turn the lights off,Turn,the,lights,off", "Lights off,Lights,off", "Volume up,Volume,up", "Turn up the volume,Turn,up,the,volume", "Turn the volume up,Turn,the,volume,up", "Make the music louder,Make,the,music,louder", "Turn down the volume,Turn,down,the,volume", "Turn the volume down,Turn,the,volume,down", "Make the music softer,Make,the,music,softer", "Turn up the temperature,Turn,up,the,temperature", "Turn the temperature up,Turn,the,temperature,up", "Turn up the heat,Turn,up,the,heat", "Turn the heat up,Turn,the,heat,up", "Turn down the temperature,Turn,down,the,temperature", "Turn the temperature down,Turn,the,temperature,down", "Turn down the heat,Turn,down,the,heat", "Turn the heat down,Turn,the,heat,down", "Turn off the music,Turn,off,the,music", "Stop the music,Stop,the,music", "Play,Play", "Put on the music,Put,on,the,music", "Play the music,Play,the,music", "Start the music,Start,the,music", "Turn on the lamp,Turn,on,the,lamp", "Switch on the lamp,Switch,on,the,lamp", "Turn the lamp on,Turn,the,lamp,on", "Lamp on,Lamp,on", "Turn off the lamp,Turn,off,the,lamp", "Turn the lamp off,Turn,the,lamp,off", "Switch off the lamp,Switch,off,the,lamp", "Lamp off,Lamp,off", "Switch the lights on,Switch,the,lights,on", "Turn on the kitchen lights,Turn,on,the,kitchen,lights", "Switch on the kitchen lights,Switch,on,the,kitchen,lights", "Turn the lights on in the kitchen,Turn,the,lights,on,in,the,kitchen", "Switch on the lights in the kitchen,Switch,on,the,lights,in,the,kitchen", "Lights on in the kitchen,Lights,on,in,the,kitchen", "Kitchen lights on,Kitchen,lights,on", "Switch the kitchen lights on,Switch,the,kitchen,lights,on", "Turn the kitchen lights on,Turn,the,kitchen,lights,on", "Turn on the lights in the bedroom,Turn,on,the,lights,in,the,bedroom", "Turn the bedroom lights on,Turn,the,bedroom,lights,on", "Switch on the lights in the bedroom,Switch,on,the,lights,in,the,bedroom", "Switch the bedroom lights on,Switch,the,bedroom,lights,on", "Lights on in the bedroom,Lights,on,in,the,bedroom", "Bedroom lights on,Bedroom,lights,on", "Turn on the washroom lights,Turn,on,the,washroom,lights", "Turn the washroom lights on,Turn,the,washroom,lights,on", "Lights on in the washroom,Lights,on,in,the,washroom", "Washroom lights on,Washroom,lights,on", "Turn on the bathroom lights,Turn,on,the,bathroom,lights", "Turn the bathroom lights on,Turn,the,bathroom,lights,on", "Switch on the bathroom lights,Switch,on,the,bathroom,lights", "Lights on in the bathroom,Lights,on,in,the,bathroom", "Bathroom lights on,Bathroom,lights,on", "Switch the lights off,Switch,the,lights,off", "Turn off the kitchen lights,Turn,off,the,kitchen,lights", "Turn the lights off in the kitchen,Turn,the,lights,off,in,the,kitchen", "Switch off the lights in the kitchen,Switch,off,the,lights,in,the,kitchen", "Switch the lights off in the kitchen,Switch,the,lights,off,in,the,kitchen", "Lights off in the kitchen,Lights,off,in,the,kitchen", "Kitchen lights off,Kitchen,lights,off", "Turn off the lights in the bedroom,Turn,off,the,lights,in,the,bedroom", "Turn the bedroom lights off,Turn,the,bedroom,lights,off", "Switch off the lights in the bedroom,Switch,off,the,lights,in,the,bedroom", "Switch the bedroom lights off,Switch,the,bedroom,lights,off", "Lights off in the bedroom,Lights,off,in,the,bedroom", "Bedroom lights off,Bedroom,lights,off", "Turn off the washroom lights,Turn,off,the,washroom,lights", "Switch off the washroom lights,Switch,off,the,washroom,lights", "Lights off in the washroom,Lights,off,in,the,washroom", "Washroom lights off,Washroom,lights,off", "Turn volume up,Turn,volume,up", "Turn sound up,Turn,sound,up", "Turn it up,Turn,it,up", "Make it louder,Make,it,louder", "Louder,Louder", "Volume max,Volume,max", "Louder please,Louder,please", "Increase the volume,Increase,the,volume", "Increase the sound volume,Increase,the,sound,volume", "I need volume,I,need,volume", "Increase the sound,Increase,the,sound", "Louder phone,Louder,phone", "I can't hear that,I,ca,n't,hear,that", "Too quiet,Too,quiet", "That's too quiet,That,'s,too,quiet", "Far too quiet,Far,too,quiet", "Turn the sound up,Turn,the,sound,up", "I need to hear this, increase the volume,I,need,to,hear,this,,,increase,the,volume", "I couldn't hear anything, turn up the volume,I,could,n't,hear,anything,,,turn,up,the,volume", "This video sound is too low, turn up the volume,This,video,sound,is,too,low,,,turn,up,the,volume", "Decrease volume,Decrease,volume", "Turn volume down,Turn,volume,down", "Turn sound down,Turn,sound,down", "Decrease audio volume,Decrease,audio,volume", "Reduce audio volume,Reduce,audio,volume", "Turn it down,Turn,it,down", "Quieter,Quieter", "Volume mute,Volume,mute", "Lower the volume,Lower,the,volume", "Volume lower,Volume,lower", "Decrease the volume,Decrease,the,volume", "Decrease sound levels,Decrease,sound,levels", "Too loud,Too,loud", "That’s too loud,That,’s,too,loud", "Far too loud,Far,too,loud", "Make it quieter,Make,it,quieter", "It’s too loud, turn it down,It,’s,too,loud,,,turn,it,down", "It’s too loud, turn the volume down,It,’s,too,loud,,,turn,the,volume,down", "Increase the temperature,Increase,the,temperature", "Increase the heating,Increase,the,heating", "Could you increase the heating?,Could,you,increase,the,heating,?", "Could you increase the heating please?,Could,you,increase,the,heating,please,?", "Make it hotter,Make,it,hotter", "More heat,More,heat", "Heat up,Heat,up", "Turn up the temperature in the kitchen,Turn,up,the,temperature,in,the,kitchen", "Turn the kitchen temperature up,Turn,the,kitchen,temperature,up", "Increase the temperature in the kitchen,Increase,the,temperature,in,the,kitchen", "Increase the heating in the kitchen,Increase,the,heating,in,the,kitchen", "Turn up the heat in the kitchen,Turn,up,the,heat,in,the,kitchen", "Turn the heat up in the kitchen,Turn,the,heat,up,in,the,kitchen", "Kitchen heat up,Kitchen,heat,up", "Turn up the temperature in the bedroom,Turn,up,the,temperature,in,the,bedroom", "Turn the temperature in the bedroom up,Turn,the,temperature,in,the,bedroom,up", "Turn up the bedroom heat,Turn,up,the,bedroom,heat", "Turn the bedroom heat up,Turn,the,bedroom,heat,up", "Bedroom heat up,Bedroom,heat,up", "Increase the temperature in the bedroom,Increase,the,temperature,in,the,bedroom", "Increase the heating in the bedroom,Increase,the,heating,in,the,bedroom", "Turn up the washroom temperature,Turn,up,the,washroom,temperature", "Turn the temperature up in the washroom,Turn,the,temperature,up,in,the,washroom", "Turn up the heat in the washroom,Turn,up,the,heat,in,the,washroom", "Turn the heat up in the washroom,Turn,the,heat,up,in,the,washroom", "Washroom heat up,Washroom,heat,up", "Increase the temperature in the washroom,Increase,the,temperature,in,the,washroom", "Increase the heating in the washroom,Increase,the,heating,in,the,washroom", "Turn up the bathroom temperature,Turn,up,the,bathroom,temperature", "Turn the temperature up in the bathroom,Turn,the,temperature,up,in,the,bathroom", "Turn up the heat in the bathroom,Turn,up,the,heat,in,the,bathroom", "Turn the heat up in the bathroom,Turn,the,heat,up,in,the,bathroom", "Bathroom heat up,Bathroom,heat,up", "Increase the temperature in the bathroom,Increase,the,temperature,in,the,bathroom", "Increase the heating in the bathroom,Increase,the,heating,in,the,bathroom", "Make it cooler,Make,it,cooler", "Less heat,Less,heat", "Heat down,Heat,down", "Decrease the temperature,Decrease,the,temperature", "Decrease the heating,Decrease,the,heating", "Could you decrease the heating?,Could,you,decrease,the,heating,?", "Could you decrease the heating please?,Could,you,decrease,the,heating,please,?", "Turn down the temperature in the kitchen,Turn,down,the,temperature,in,the,kitchen", "Turn the kitchen temperature down,Turn,the,kitchen,temperature,down", "Turn down the heat in the kitchen,Turn,down,the,heat,in,the,kitchen", "Turn the heat down in the kitchen,Turn,the,heat,down,in,the,kitchen", "Kitchen heat down,Kitchen,heat,down", "Decrease the temperature in the kitchen,Decrease,the,temperature,in,the,kitchen", "Decrease the heating in the kitchen,Decrease,the,heating,in,the,kitchen", "Turn down the temperature in the bedroom,Turn,down,the,temperature,in,the,bedroom", "Turn the temperature in the bedroom down,Turn,the,temperature,in,the,bedroom,down", "Turn down the bedroom heat,Turn,down,the,bedroom,heat", "Turn the bedroom heat down,Turn,the,bedroom,heat,down", "Bedroom heat down,Bedroom,heat,down", "Decrease the temperature in the bedroom,Decrease,the,temperature,in,the,bedroom", "Decrease the heating in the bedroom,Decrease,the,heating,in,the,bedroom", "Turn down the washroom temperature,Turn,down,the,washroom,temperature", "Turn the temperature down in the washroom,Turn,the,temperature,down,in,the,washroom", "Turn down the heat in the washroom,Turn,down,the,heat,in,the,washroom", "Turn the heat down in the washroom,Turn,the,heat,down,in,the,washroom", "Decrease the temperature in the washroom,Decrease,the,temperature,in,the,washroom", "Decrease the heating in the washroom,Decrease,the,heating,in,the,washroom", "Washroom heat down,Washroom,heat,down", "Turn down the bathroom temperature,Turn,down,the,bathroom,temperature", "Turn the temperature down in the bathroom,Turn,the,temperature,down,in,the,bathroom", "Turn down the heat in the bathroom,Turn,down,the,heat,in,the,bathroom", "Turn the heat down in the bathroom,Turn,the,heat,down,in,the,bathroom", "Decrease the temperature in the bathroom,Decrease,the,temperature,in,the,bathroom", "Decrease the heating in the bathroom,Decrease,the,heating,in,the,bathroom", "Bathroom heat down,Bathroom,heat,down", "Pause music,Pause,music", "Stop music,Stop,music", "Stop,Stop", "Play music,Play,music", "Resume music,Resume,music", "Bring me the newspaper,Bring,me,the,newspaper", "Get me the newspaper,Get,me,the,newspaper", "Bring the newspaper,Bring,the,newspaper", "Bring newspaper,Bring,newspaper", "Go get the newspaper,Go,get,the,newspaper", "Fetch the newspaper,Fetch,the,newspaper", "Bring me some juice,Bring,me,some,juice", "Get me some juice,Get,me,some,juice", "Bring some juice,Bring,some,juice", "Bring juice,Bring,juice", "Go get me some juice,Go,get,me,some,juice", "Bring me my socks,Bring,me,my,socks", "Get me my socks,Get,me,my,socks", "Bring socks,Bring,socks", "Go get me my socks,Go,get,me,my,socks", "Fetch my socks,Fetch,my,socks", "Bring me my shoes,Bring,me,my,shoes", "Get me my shoes,Get,me,my,shoes", "Bring shoes,Bring,shoes", "Bring my shoes,Bring,my,shoes", "Go get me my shoes,Go,get,me,my,shoes", "Fetch my shoes,Fetch,my,shoes", "Allow a different language,Allow,a,different,language", "Use a different language,Use,a,different,language", "Change the language,Change,the,language", "Switch the language,Switch,the,language", "Switch language,Switch,language", "Switch languages,Switch,languages", "Change system language,Change,system,language", "Open language settings,Open,language,settings", "Set the language,Set,the,language", "Set language to Chinese,Set,language,to,Chinese", "Set language to German,Set,language,to,German", "Set language to Korean,Set,language,to,Korean", "Set language to English,Set,language,to,English", "Set my device to Chinese,Set,my,device,to,Chinese", "OK now switch the main language to Chinese,OK,now,switch,the,main,language,to,Chinese", "OK now switch the main language to German,OK,now,switch,the,main,language,to,German", "OK now switch the main language to Korean,OK,now,switch,the,main,language,to,Korean", "OK now switch the main language to English,OK,now,switch,the,main,language,to,English", "Set my phone's language to Chinese,Set,my,phone,'s,language,to,Chinese", "Set my phone's language to German,Set,my,phone,'s,language,to,German", "Set my phone's language to Korean,Set,my,phone,'s,language,to,Korean", "Set my phone's language to English,Set,my,phone,'s,language,to,English", "I need to practice my Chinese. Switch the language,I,need,to,practice,my,Chinese,.,Switch,the,language", "I need to practice my German. Switch the language,I,need,to,practice,my,German,.,Switch,the,language", "I need to practice my Korean. Switch the language,I,need,to,practice,my,Korean,.,Switch,the,language", "I need to practice my English. Switch the language,I,need,to,practice,my,English,.,Switch,the,language", "Turn the washroom lights off,Turn,the,washroom,lights,off", "Pause,Pause", "Switch on the washroom lights,Switch,on,the,washroom,lights", "Bring my socks,Bring,my,socks", "Language settings,Language,settings", "Turn lights on,Turn,lights,on", "Turn light on,Turn,light,on", "Turn light off,Turn,light,off", "Turn lights off,Turn,lights,off", "Switch lights on,Switch,lights,on", "Switch light on,Switch,light,on", "Switch light off,Switch,light,off", "Switch lights off,Switch,lights,off", "Decrease temperature,Decrease,temperature", "Increase temperature,Increase,temperature", "Decrease heating,Decrease,heating", "Increase heating,Increase,heating", "Fetch newspaper,Fetch,newspaper", "Get newspaper,Get,newspaper", "Get shoes,Get,shoes", "Can't hear,Ca,n't,hear", "Cannot hear,Can,not,hear", "Mute sound,Mute,sound", "Mute volume,Mute,volume", "Kitchen temperature down,Kitchen,temperature,down", "Kitchen temperature up,Kitchen,temperature,up", "Bedroom temperature down,Bedroom,temperature,down", "Bedroom temperature up,Bedroom,temperature,up", "Switch language to Chinese,Switch,language,to,Chinese", "Switch language to German,Switch,language,to,German", "Switch language to Korean,Switch,language,to,Korean", "Switch language to English,Switch,language,to,English"]
# Change labels as per the words to record - copy paste from label_generator_for_record_serial.py
labels = generate_labels(len(word_map))
#labels = [174, 150, 106, 5, 116, 114, 247, 184, 39, 26, 98, 72, 243, 168, 43, 244, 201, 187, 181, 229, 65, 179, 4, 105, 231, 194, 192, 154, 21, 103, 89, 145, 153, 246, 156, 173, 135, 6, 80, 228, 208, 15, 22, 12, 9, 63, 175, 139, 163, 210, 149, 120, 52, 226, 142, 87, 54, 164, 78, 7, 199, 108, 220, 20, 10, 238, 14, 0, 58, 95, 240, 16, 189, 221, 112, 191, 2, 242, 209, 161, 178, 157, 96, 211, 172, 41, 101, 77, 83, 104, 29, 147, 122, 88, 51, 37, 60, 148, 42, 207, 180, 223, 33, 17, 185, 53, 234, 196, 23, 55, 73, 205, 86, 115, 237, 130, 182, 25, 117, 171, 24, 128, 79, 18, 38, 158, 70, 188, 127, 61, 143, 245, 97, 82, 100, 19, 44, 133, 109, 1, 102, 111, 144, 215, 57, 224, 40, 176, 230, 113, 170, 125, 76, 162, 140, 190, 94, 126, 218, 71, 195, 118, 50, 212, 204, 134, 132, 136, 62, 107, 34, 169, 222, 214, 177, 129, 74, 225, 198, 206, 11, 239, 3, 92, 59, 193, 36, 183, 66, 49, 197, 227, 151, 233, 216, 166, 186, 56, 69, 165, 121, 84, 32, 93, 219, 67, 217, 27, 202, 203, 48, 236, 31, 124, 131, 167, 152, 35, 81, 119, 64, 46, 30, 146, 13, 123, 232, 91, 200, 155, 110, 213, 8, 85, 141, 241, 138, 47, 99, 45, 90, 137, 159, 68, 75, 28, 160, 235]

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
        # print 'WPM:', 60.0 / (float(recorded_length) / 250 / len(word_map[labels[recorded_count-(1 if end < len(trigger_history)-1 else 0)]].split(' ')))
        print 'WPM:', 60.0 / (float(recorded_length) / 250)
    print
    print 'Sample #' + str(recorded_count + 1) + '/' + str(len(labels) - 1), '\tNext:', word_map[labels[recorded_count]]
    print


preprocessing.serial.start('/dev/tty.usbserial-DM01HUN9',
                  on_data, channels=channels, transform_fn=transform_data,
                  history_size=2500, shown_size=1200, override_step=100, bipolar=False)  # 35