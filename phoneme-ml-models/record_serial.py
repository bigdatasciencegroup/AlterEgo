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

import data

channels = range(0, 8) # Must be same as trained model if test_model==True
#channels = range(0, 4) # Must be same as trained model if test_model==True
#channels = range(0, 3) # Must be same as trained model if test_model==True
#channels = range(0, 1) # Must be same as trained model if test_model==True
#channels = range(1, 8) # Must be same as trained model if test_model==True
#channels = [1, 3, 4] # DO NOT CHANGE

def transform_data(sequence_groups, sample_rate=250):
    #### Apply DC offset and drift correction
    drift_low_freq = 0.5 #0.5
    sequence_groups = data.transform.subtract_initial(sequence_groups)
    sequence_groups = data.transform.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    sequence_groups = data.transform.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq
    notch_freq = 60
    num_times = 3 #pretty much just the filter order
    freqs = map(int, map(round, np.arange(1, sample_rate/(2. * notch_freq)) * notch_freq))
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = data.transform.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    #sequence_groups = data.transform.normalize_std(sequence_groups)

    def normalize_kernel(kernel, subtract_mean=False):
        if subtract_mean:
            kernel = np.array(kernel, np.float32) - np.mean(kernel)
        return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
    def ricker_function(t, sigma):
        return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
    def ricker_wavelet(n, sigma):
        return np.array(map(lambda x: ricker_function(x, sigma), range(-n//2, n//2+1)))

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = data.transform.correlate(sequence_groups, ricker_kernel)
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
    sequence_groups = data.transform.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)
    
    #### Apply hard bandpassing
#    sequence_groups = data.transform.fft(sequence_groups)
#    sequence_groups = data.transform.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
#    sequence_groups = np.real(data.transform.ifft(sequence_groups))
    
    return sequence_groups

'''
# Common words phonemes dataset 1
word_map = ['aches EY K S', 'afternoon AE F T ER N UW N', 'allow AH L AW', 'am AE M', 'anything EH N IY TH IH NG', 'audio AA D IY OW', 'back B AE K', 'backache B AE K EY K', 'bad B AE D', 'bathroom B AE TH R UW M', 'bed B EH D', 'bedroom B EH D R UW M', 'bring B R IH NG', 'bye B AY', 'can K AE N', 'cant K AE N T', 'cannot K AE N AA T', 'change CH EY N JH', 'chinese CH AY N IY Z', 'computer K AH M P Y UW T ER', 'cooler K UW L ER', 'could K UH D', 'couldnt K UH D AH N T', 'decrease D IH K R IY S', 'device D IH V AY S', 'different D IH F ER AH N T', 'down D AW N', 'drink D R IH NG K', 'eat IY T', 'english IH NG G L IH SH', 'evening IY V N IH NG', 'exhausted IH G Z AO S T IH D', 'far F AA R', 'feel F IY L', 'fetch F EH CH', 'german JH ER M AH N', 'get G EH T', 'getting G EH T IH NG', 'go G OW', 'good G UH D', 'goodbye G UH D B AY', 'have HH AE V', 'having HH AE V IH NG', 'head HH EH D', 'headache HH EH D EY K', 'hear HH IY R', 'heat HH IY T', 'heating HH IY T IH NG', 'hello HH AH L OW', 'help HH EH L P', 'hey HH EY', 'hi HH AY', 'hotter HH AA T ER', 'hungry HH AH NG G R IY', 'hurting HH ER T IH NG', 'hurts HH ER T S', 'in IH N', 'increase IH N K R IY S', 'into IH N T UW', 'is IH Z', 'it IH T', 'its IH T S', 'juice JH UW S', 'kitchen K IH CH AH N', 'korean K AO R IY AH N', 'lamp L AE M P', 'language L AE NG G W AH JH', 'languages L AE NG G W AH JH AH Z', 'later L EY T ER', 'less L EH S', 'levels L EH V AH L Z', 'lie L AY', 'light L AY T', 'lights L AY T S', 'lot L AA T', 'loud L AW D', 'louder L AW D ER', 'low L OW', 'lower L OW ER', 'main M EY N', 'make M EY K', 'max M AE K S', 'me M IY', 'more M AO R', 'morning M AO R N IH NG', 'much M AH CH', 'music M Y UW Z IH K', 'mute M Y UW T', 'my M AY', 'nap N AE P', 'need N IY D', 'newspaper N UW Z P EY P ER', 'now N AW', 'of AH V', 'off AO F', 'ok OW K EY', 'on AA N', 'open OW P AH N', 'pain P EY N', 'pains P EY N Z', 'pause P AO Z', 'phone F OW N', 'phones F OW N Z', 'play P L EY', 'please P L IY Z', 'practice P R AE K T IH S', 'put P UH T', 'quiet K W AY AH T', 'quieter K W AY AH T ER', 'reboot R IY B UW T', 'reduce R IH D UW S', 'restart R IY S T AA R T', 'resume R IH Z UW M', 'see S IY', 'set S EH T', 'settings S EH T IH NG Z', 'shoes SH UW Z', 'sleep S L IY P', 'so S OW', 'socks S AA K S', 'softer S AA F T ER', 'some S AH M', 'something S AH M TH IH NG', 'soon S UW N', 'sound S AW N D', 'start S T AA R T', 'stop S T AA P', 'switch S W IH CH', 'system S IH S T AH M', 'take T EY K', 'temperature T EH M P R AH CH ER', 'thank TH AE NG K', 'thanks TH AE NG K S', 'that DH AE T', 'thats DH AE T S', 'the DH AH', 'thirsty TH ER S T IY', 'this DH IH S', 'tired T AY ER D', 'to T UW', 'too T UW', 'turn T ER N', 'tv T IY V IY', 'up AH P', 'use Y UW S', 'very V EH R IY', 'video V IH D IY OW', 'volume V AA L Y UW M', 'want W AA N T', 'washroom W AA SH R UW M', 'watch W AA CH', 'water W AO T ER', 'you Y UW', 'FINISHED FINISHED FINISHED FINISHED FINISHED']

# 1) [59, 33, 69, 127, 31, 107, 97, 14, 53, 44, 147, 138, 78, 102, 5, 85, 16, 100, 89, 95, 91, 124, 92, 99, 93, 40, 29, 42, 54, 122, 35, 19, 113, 103, 77, 145, 58, 121, 56, 28, 51, 109, 66, 17, 119, 126, 132, 73, 18, 48]
# 2) [4, 45, 75, 39, 36, 118, 141, 108, 12, 148, 84, 123, 136, 11, 114, 2, 105, 6, 0, 111, 65, 55, 104, 135, 46, 62, 74, 117, 94, 151, 90, 10, 34, 32, 125, 38, 83, 112, 152, 27, 23, 67, 9, 131, 98, 120, 143, 87, 142, 110]
# 3) [64, 15, 82, 41, 80, 52, 26, 76, 43, 24, 149, 116, 130, 49, 21, 70, 3, 146, 30, 150, 106, 47, 115, 13, 88, 8, 81, 60, 128, 1, 57, 22, 61, 63, 7, 86, 96, 68, 50, 139, 101, 20, 25, 134, 71, 129, 144, 79, 133, 137, 72, 140, 37]
'''

# Common words phonemes dataset 2
word_map = ['much M AH CH', 'obliged AH B L AY JH D', 'to T UW', 'you Y UW', 'it IH T', 'would W UH D', 'be B IY', 'ideal AY D IY L', 'if IH F', 'help HH EH L P', 'me M IY', 'get G EH T', 'up AH P', 'its IH T S', 'not N AA T', 'too T UW', 'trouble T R AH B AH L', 'farewell F EH R W EH L', 'dont D OW N T', 'mind M AY N D', 'restroom R EH S T R UW M', 'have HH AE V', 'go G OW', 'the DH AH', 'kindly K AY N D L IY', 'assist AH S IH S T', 'worn W AO R N', 'out AW T', 'so S OW', 'excessively IH K S EH S IH V L IY', 'some S AH M', 'rest R EH S T', 'need N IY D', 'snooze S N UW Z', 'sleep S L IY P', 'into IH N T UW', 'bed B EH D', 'eat IY T', 'something S AH M TH IH NG', 'drink D R IH NG K', 'water W AO T ER', 'on AA N', 'turn T ER N', 'off AO F', 'tv T IY V IY', 'pc P IY S IY', 'restart R IY S T AA R T', 'reboot R IY B UW T', 'my M AY', 'am AE M', 'in IH N', 'agony AE G AH N IY', 'ton T AH N', 'of AH V', 'pain P EY N', 'totally T OW T AH L IY', 'drained D R EY N D', 'how HH AW', 'are AA R', 'doing D UW IH NG', 'feeling F IY L IH NG', 'today T AH D EY', 'good G UH D', 'night N AY T', 'quiet K W AY AH T', 'whats W AH T S', 'heart HH AA R T', 'rate R EY T', 'what W AH T', 'is IH Z', 'take T EY K', 'meds M EH D Z', 'medications M EH D AH K EY SH AH N Z', 'remind R IY M AY N D', 'please P L IY Z', 'set S EH T', 'telephones T EH L AH F OW N Z', 'language L AE NG G W AH JH', 'korean K AO R IY AH N', 'chinese CH AY N IY Z', 'english IH NG G L IH SH', 'german JH ER M AH N', 'hear HH IY R', 'this DH IH S', 'increment IH N K R AH M AH N T', 'volume V AA L Y UW M', 'utilize Y UW T AH L AY Z', 'an AE N', 'alternate AO L T ER N AH T', 'use Y UW S', 'other AH DH ER', 'allow AH L AW', 'permit P ER M IH T', 'loud L AW D ', 'down D AW N', 'couldnt K UH D AH N T', 'anything EH N IY TH IH NG', 'increase IH N K R IY S', 'paper P EY P ER', 'temperature T EH M P R AH CH ER', 'kitchen K IH CH AH N', 'heating HH IY T IH NG', 'bedroom B EH D R UW M', 'cold K OW L D', 'heat HH IY T', 'hot HH AA T', 'play P L EY', 'song S AO NG', 'music M Y UW Z IH K', 'next N EH K S T', 'start S T AA R T', 'playing P L EY IH NG', 'white W AY T', 'forest F AO R AH S T', 'sounds S AW N D Z', 'rain R EY N', 'indie IH N D IY', 'pop P AA P', 'ambient AE M B IY AH N T', 'dance D AE N S', 'playlist P L EY L IH S T', 'favourite F EY V ER IH T', 'switch S W IH CH', 'change CH EY N JH', 'yes Y EH S', 'no N OW', 'thank TH AE NG K', 'away AH W EY', 'leave L IY V', 'alone AH L OW N', 'happy HH AE P IY', 'birthday B ER TH D EY', 'when W EH N', 'your Y AO R', 'time T AY M', 'london L AH N D AH N', 'geneva JH AH N IY V AH', 'new N UW', 'york Y AO R K', 'mexico M EH K S AH K OW', 'city S IH T IY', 'raise R EY Z', 'one W AH N', 'degree D IH G R IY', 'two T UW', 'degrees D IH G R IY Z', 'three TH R IY', 'lower L OW ER', 'bathroom B AE TH R UW M', 'cooling K UW L IH NG', 'all AO L', 'lights L AY T S', 'inside IH N S AY D', 'outside AW T S AY D', 'hey HH EY', 'there DH EH R', 'hello HH AH L OW', 'alarm AH L AA R M', 'cancel K AE N S AH L', 'agenda AH JH EH N D AH', 'reminder R IY M AY N D ER', 'reminders R IY M AY N D ER Z', 'tomorrow T AH M AA R OW', 'delete D IH L IY T', 'for F AO R', 'daily D EY L IY', 'brief B R IY F', 'timer T AY M ER', 'left L EH F T', 'tell T EH L', 'news N UW Z', 'weather W EH DH ER', 'joke JH OW K', 'hows HH AW Z', 'do D UW', 'umbrella AH M B R EH L AH', 'show SH OW', 'phone F OW N', 'bedtime B EH D T AY M', 'story S T AO R IY', 'flight F L AY T', 'todays T AH D EY Z', 'warm W AO R M', 'day D EY', 'meeting M IY T IH NG', 'first F ER S T', 'which W IH CH', 'thanks TH AE NG K S', 'helping HH EH L P IH NG', 'events IH V EH N T S', 'calendar K AE L AH N D ER', 'call K AO L', 'doctor D AA K T ER', 'nurse N ER S', 'pause P AO Z', 'resume R IH Z UW M', 'make M EY K', 'louder L AW D ER', 'softer S AA F T ER', 'stop S T AA P', 'put P UH T', 'lamp L AE M P', 'washroom W AA SH R UW M', 'sound S AW N D', 'max M AE K S', 'cant K AE N T', 'that DH AE T', 'thats DH AE T S', 'far F AA R', 'video V IH D IY OW', 'low L OW', 'decrease D IH K R IY S', 'audio AA D IY OW', 'reduce R IH D UW S', 'quieter K W AY AH T ER', 'mute M Y UW T', 'levels L EH V AH L Z', 'could K UH D', 'hotter HH AA T ER', 'more M AO R', 'cooler K UW L ER', 'less L EH S', 'bring B R IH NG', 'newspaper N UW Z P EY P ER', 'fetch F EH CH', 'juice JH UW S', 'socks S AA K S', 'shoes SH UW Z', 'different D IH F ER AH N T', 'languages L AE NG G W AH JH AH Z', 'system S IH S T AH M', 'open OW P AH N', 'settings S EH T IH NG Z', 'device D IH V AY S', 'ok OW K EY', 'now N AW', 'main M EY N', 'phones F OW N Z', 'practice P R AE K T IH S', 'light L AY T', 'cannot K AE N AA T', 'hi HH AY', 'morning M AO R N IH NG', 'evening IY V N IH NG', 'afternoon AE F T ER N UW N', 'goodbye G UH D B AY', 'bye B AY', 'see S IY', 'soon S UW N', 'later L EY T ER', 'can K AE N', 'tired T AY ER D', 'nap N AE P', 'want W AA N T', 'lie L AY', 'headache HH EH D EY K', 'head HH EH D', 'hurts HH ER T S', 'hungry HH AH NG G R IY', 'thirsty TH ER S T IY', 'backache B AE K EY K', 'back B AE K', 'aches EY K S', 'watch W AA CH', 'feel F IY L', 'computer K AH M P Y UW T ER', 'very V EH R IY', 'lot L AA T', 'exhausted IH G Z AO S T IH D', 'having HH AE V IH NG', 'getting G EH T IH NG', 'bad B AE D', 'hurting HH ER T IH NG', 'pains P EY N Z', 'adventure AE D V EH N CH ER', 'actual AE K CH AH W AH L', 'algorithm AE L G ER IH DH AH M', 'although AO L DH OW', 'another AH N AH DH ER', 'budget B AH JH IH T', 'challenge CH AE L AH N JH', 'fellowship F EH L OW SH IH P', 'fertilization F ER T AH L IH Z EY SH AH N', 'fiction F IH K SH AH N', 'garnish G AA R N IH SH', 'girlish G ER L IH SH', 'good G UH D', 'hook HH UH K', 'plural P L UH R AH L', 'premature P R IY M AH CH UH R', 'procure P R OW K Y UH R', 'wavelength W EY V L EH NG TH', 'wealth W EH L TH', 'with W IH TH', 'FINISHED FINISHED FINISHED FINISHED FINISHED']

# 1) [268, 269, 280, 110, 167, 117, 147, 273, 182, 27, 191, 95, 232, 11, 225, 283, 12, 179, 18, 170, 107, 119, 91, 99, 220, 180, 159, 89, 223, 206, 293, 245, 175, 58, 250, 62, 260, 120, 284, 73, 127, 59, 177, 266, 85, 51, 158, 204, 185, 150]
# 2) [90, 70, 230, 292, 187, 240, 39, 291, 102, 289, 236, 278, 4, 17, 88, 42, 244, 122, 5, 112, 38, 205, 123, 181, 146, 34, 173, 168, 131, 189, 274, 19, 14, 192, 176, 44, 157, 139, 108, 248, 251, 79, 277, 132, 253, 172, 105, 8, 243, 174]
# 3) [239, 31, 163, 55, 234, 224, 161, 48, 227, 211, 201, 33, 93, 138, 35, 247, 63, 29, 0, 213, 78, 66, 287, 67, 286, 106, 28, 16, 219, 188, 47, 208, 197, 40, 21, 101, 69, 53, 137, 24, 134, 194, 116, 154, 142, 218, 217, 56, 61, 164]
# 4) [229, 84, 169, 114, 265, 238, 118, 193, 145, 288, 249, 54, 261, 171, 186, 184, 98, 214, 130, 97, 279, 82, 282, 60, 94, 140, 212, 148, 152, 263, 242, 81, 124, 221, 222, 13, 96, 228, 45, 202, 103, 36, 262, 20, 258, 200, 75, 207, 271, 149]
# 5) [257, 2, 198, 290, 6, 128, 259, 77, 113, 65, 183, 151, 160, 46, 190, 74, 92, 199, 246, 87, 143, 233, 10, 162, 32, 136, 166, 83, 57, 165, 155, 100, 125, 23, 126, 231, 195, 9, 270, 104, 153, 256, 135, 275, 226, 111, 25, 196, 64, 15]
# 6) [210, 41, 267, 109, 80, 52, 26, 76, 43, 3, 49, 285, 30, 121, 115, 272, 216, 264, 209, 1, 22, 7, 141, 86, 241, 215, 68, 50, 156, 252, 254, 276, 178, 281, 237, 71, 129, 144, 133, 203, 255, 72, 235, 37]


labels = [268, 269, 280, 110, 167, 117, 147, 273, 182, 27, 191, 95, 232, 11, 225, 283, 12, 179, 18, 170, 107, 119, 91, 99, 220, 180, 159, 89, 223, 206, 293, 245, 175, 58, 250, 62, 260, 120, 284, 73, 127, 59, 177, 266, 85, 51, 158, 204, 185, 150]

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
    print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i+1) for i in range(8)])
    print str('{:.1f}'.format(count/250.)) + 's\t\t' + '\t'.join(
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
        # print 'WPM:', 60.0 / (float(recorded_length) / 250)
    print
    print 'Sample #' + str(recorded_count+1)+'/'+str(len(labels)-1), '\tNext:', word_map[labels[recorded_count]]
    print

# old: DM01HUN9 
# new: DM01HQ99
data.serial.start('/dev/tty.usbserial-DM01HQ99',
                  on_data, channels=channels, transform_fn=transform_data,
                  history_size=2500, shown_size=1200, override_step=100, bipolar=False)#35