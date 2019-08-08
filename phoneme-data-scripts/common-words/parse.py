''' 
Ran using Python 3.7.3

Creates dataset of 5 phoneme words by pruning cmudict-0_7b:
1) 29 out of 39 phonemes used - Phonemes (6) with visible movements and outliers (4) removed
2) bunch of 4 phonemes in a word repeating have been excluded (first 4 positions as well as last four positions)

Writes the generated dataset (4757 words and the corresponding phonemes) to 5words.txt
Writes the word_map variable (in record_serial.py) value - [value] - to 5words_for_wordmap.txt
Prints (on terminal) out the distribution of phonemes in the generated dataset and training dataset 
'''

import os
import numpy as np
from nltk.util import ngrams
from pandas import DataFrame

lines = [line.rstrip('\n')\
 for line in open('common_words_phonemes_2.txt', 'r', encoding='ISO-8859-1')]

data = []

# with open('wordmap_2.txt', 'r') as file:
# 	filedata = file.read()

# filedata = filedata.replace('\n', '\', \'')

# open('wordmap_2.txt', 'w+').close() # To erase all previous contents of the file
# with open('wordmap_2.txt', 'w') as file:
# 	file.write(filedata)


for line in lines:
	data.append(line.split())

# data = np.asarray(data)

# print(data)

# total 39 phonemes
AA,AE,AH,AO,AW,AY,EH,ER,EY,IH,IY,OW,OY,UH,UW = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 # 15
CH,D,DH,G,HH,JH,K,L,N,NG,R,S,SH,T,TH,Y,Z,ZH,P,B,F,M,V,W = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 # 24

for x in data:
	for y in x:
		if y=='AA':		AA+=1
		elif y=='AE':	AE+=1
		elif y=='AH':	AH+=1
		elif y=='AO':	AO+=1
		elif y=='AW':	AW+=1
		elif y=='AY':	AY+=1
		elif y=='EH':	EH+=1
		elif y=='ER':	ER+=1
		elif y=='EY':	EY+=1
		elif y=='IH':	IH+=1
		elif y=='IY':	IY+=1
		elif y=='OW':	OW+=1
		elif y=='OY':	OY+=1
		elif y=='UH':	UH+=1
		elif y=='UW':	UW+=1

		elif y=='CH':	CH+=1
		elif y=='D':	D+=1
		elif y=='DH':	DH+=1
		elif y=='G':	G+=1
		elif y=='HH':	HH+=1
		elif y=='JH':	JH+=1
		elif y=='K':	K+=1
		elif y=='L':	L+=1
		elif y=='N':	N+=1
		elif y=='NG':	NG+=1
		elif y=='R':	R+=1
		elif y=='S':	S+=1
		elif y=='SH':	SH+=1
		elif y=='T':	T+=1
		elif y=='TH':	TH+=1
		elif y=='Y':	Y+=1
		elif y=='Z':	Z+=1
		elif y=='ZH':	ZH+=1
		elif y=='P':	P+=1
		elif y=='B':	B+=1
		elif y=='F':	F+=1
		elif y=='M':	M+=1
		elif y=='V':	V+=1
		elif y=='W':	W+=1
		
print('\nDistribution in entire generated dataset')
print('AA :',AA, '  AE :',AE, '  AH :',AH, '  AO :',AO, '  AW :',AW, '  AY :',AY, '  EH :',EH, '  ER :',ER, '  EY :',EY, '  IH :',IH, '  IY :',IY, '  OW :',OW, ' OY :',OY, ' UH :',UH, '  UW :',UW)
print('CH :',CH, '  D :',D, ' DH :',DH, '  G :',G, '  HH :',HH, '  JH :',JH, '  K :',K, '  L :',L, '  N :',N, '  NG :',NG, '  R :',R, '  S :',S, '  SH :',SH, '  T :',T, '  TH :',TH, '  Y :',Y, '  Z :',Z, ' ZH :',ZH, ' P :',P, ' B :',P, ' M :',M, ' F:',F, ' V :',V, ' W:',W)
print('Total Phonemes :', AA+AE+AH+AO+AW+AY+EH+ER+EY+IH+IY+OW+OY+UH+UW+CH+D+DH+G+HH+JH+K+L+N+NG+R+S+SH+T+TH+Y+Z+ZH+P+B+M+V+W+F)		

'''
Distribution in entire generated dataset 1 (153 words)
AA : 15   AE : 21   AH : 24   AO : 7   AW : 6   AY : 12   EH : 15   ER : 19   EY : 14   IH : 33   IY : 25   OW : 11  OY : 0  UH : 5   UW : 21
CH : 8   D : 23  DH : 4   G : 10   HH : 15   JH : 5   K : 28   L : 25   N : 34   NG : 17   R : 22   S : 36   SH : 3   T : 49   TH : 6   Y : 6   Z : 14  ZH : 0  P : 18  B : 18  M : 23  F: 9  V : 10  W: 9
Total Phonemes : 612

Distribution in entire generated dataset 2 (294 words)
AA : 24   AE : 32   AH : 73   AO : 19   AW : 11   AY : 28   EH : 40   ER : 38   EY : 31   IH : 63   IY : 52   OW : 24  OY : 0  UH : 11   UW : 30
CH : 13   D : 59  DH : 10   G : 17   HH : 22   JH : 11   K : 47   L : 67   N : 80   NG : 25   R : 58   S : 61   SH : 10   T : 97   TH : 11   Y : 11   Z : 27  ZH : 0  P : 32  B : 32  M : 44  F: 23  V : 17  W: 25
Total Phonemes : 1262
'''

word_map = ['much M AH CH', 'obliged AH B L AY JH D', 'to T UW', 'you Y UW', 'it IH T', 'would W UH D', 'be B IY', 'ideal AY D IY L', 'if IH F', 'help HH EH L P', 'me M IY', 'get G EH T', 'up AH P', 'its IH T S', 'not N AA T', 'too T UW', 'trouble T R AH B AH L', 'farewell F EH R W EH L', 'dont D OW N T', 'mind M AY N D', 'restroom R EH S T R UW M', 'have HH AE V', 'go G OW', 'the DH AH', 'kindly K AY N D L IY', 'assist AH S IH S T', 'worn W AO R N', 'out AW T', 'so S OW', 'excessively IH K S EH S IH V L IY', 'some S AH M', 'rest R EH S T', 'need N IY D', 'snooze S N UW Z', 'sleep S L IY P', 'into IH N T UW', 'bed B EH D', 'eat IY T', 'something S AH M TH IH NG', 'drink D R IH NG K', 'water W AO T ER', 'on AA N', 'turn T ER N', 'off AO F', 'tv T IY V IY', 'pc P IY S IY', 'restart R IY S T AA R T', 'reboot R IY B UW T', 'my M AY', 'am AE M', 'in IH N', 'agony AE G AH N IY', 'ton T AH N', 'of AH V', 'pain P EY N', 'totally T OW T AH L IY', 'drained D R EY N D', 'how HH AW', 'are AA R', 'doing D UW IH NG', 'feeling F IY L IH NG', 'today T AH D EY', 'good G UH D', 'night N AY T', 'quiet K W AY AH T', 'whats W AH T S', 'heart HH AA R T', 'rate R EY T', 'what W AH T', 'is IH Z', 'take T EY K', 'meds M EH D Z', 'medications M EH D AH K EY SH AH N Z', 'remind R IY M AY N D', 'please P L IY Z', 'set S EH T', 'telephones T EH L AH F OW N Z', 'language L AE NG G W AH JH', 'korean K AO R IY AH N', 'chinese CH AY N IY Z', 'english IH NG G L IH SH', 'german JH ER M AH N', 'hear HH IY R', 'this DH IH S', 'increment IH N K R AH M AH N T', 'volume V AA L Y UW M', 'utilize Y UW T AH L AY Z', 'an AE N', 'alternate AO L T ER N AH T', 'use Y UW S', 'other AH DH ER', 'allow AH L AW', 'permit P ER M IH T', 'loud L AW D ', 'down D AW N', 'couldnt K UH D AH N T', 'anything EH N IY TH IH NG', 'increase IH N K R IY S', 'paper P EY P ER', 'temperature T EH M P R AH CH ER', 'kitchen K IH CH AH N', 'heating HH IY T IH NG', 'bedroom B EH D R UW M', 'cold K OW L D', 'heat HH IY T', 'hot HH AA T', 'play P L EY', 'song S AO NG', 'music M Y UW Z IH K', 'next N EH K S T', 'start S T AA R T', 'playing P L EY IH NG', 'white W AY T', 'forest F AO R AH S T', 'sounds S AW N D Z', 'rain R EY N', 'indie IH N D IY', 'pop P AA P', 'ambient AE M B IY AH N T', 'dance D AE N S', 'playlist P L EY L IH S T', 'favourite F EY V ER IH T', 'switch S W IH CH', 'change CH EY N JH', 'yes Y EH S', 'no N OW', 'thank TH AE NG K', 'away AH W EY', 'leave L IY V', 'alone AH L OW N', 'happy HH AE P IY', 'birthday B ER TH D EY', 'when W EH N', 'your Y AO R', 'time T AY M', 'london L AH N D AH N', 'geneva JH AH N IY V AH', 'new N UW', 'york Y AO R K', 'mexico M EH K S AH K OW', 'city S IH T IY', 'raise R EY Z', 'one W AH N', 'degree D IH G R IY', 'two T UW', 'degrees D IH G R IY Z', 'three TH R IY', 'lower L OW ER', 'bathroom B AE TH R UW M', 'cooling K UW L IH NG', 'all AO L', 'lights L AY T S', 'inside IH N S AY D', 'outside AW T S AY D', 'hey HH EY', 'there DH EH R', 'hello HH AH L OW', 'alarm AH L AA R M', 'cancel K AE N S AH L', 'agenda AH JH EH N D AH', 'reminder R IY M AY N D ER', 'reminders R IY M AY N D ER Z', 'tomorrow T AH M AA R OW', 'delete D IH L IY T', 'for F AO R', 'daily D EY L IY', 'brief B R IY F', 'timer T AY M ER', 'left L EH F T', 'tell T EH L', 'news N UW Z', 'weather W EH DH ER', 'joke JH OW K', 'hows HH AW Z', 'do D UW', 'umbrella AH M B R EH L AH', 'show SH OW', 'phone F OW N', 'bedtime B EH D T AY M', 'story S T AO R IY', 'flight F L AY T', 'todays T AH D EY Z', 'warm W AO R M', 'day D EY', 'meeting M IY T IH NG', 'first F ER S T', 'which W IH CH', 'thanks TH AE NG K S', 'helping HH EH L P IH NG', 'events IH V EH N T S', 'calendar K AE L AH N D ER', 'call K AO L', 'doctor D AA K T ER', 'nurse N ER S', 'pause P AO Z', 'resume R IH Z UW M', 'make M EY K', 'louder L AW D ER', 'softer S AA F T ER', 'stop S T AA P', 'put P UH T', 'lamp L AE M P', 'washroom W AA SH R UW M', 'sound S AW N D', 'max M AE K S', 'cant K AE N T', 'that DH AE T', 'thats DH AE T S', 'far F AA R', 'video V IH D IY OW', 'low L OW', 'decrease D IH K R IY S', 'audio AA D IY OW', 'reduce R IH D UW S', 'quieter K W AY AH T ER', 'mute M Y UW T', 'levels L EH V AH L Z', 'could K UH D', 'hotter HH AA T ER', 'more M AO R', 'cooler K UW L ER', 'less L EH S', 'bring B R IH NG', 'newspaper N UW Z P EY P ER', 'fetch F EH CH', 'juice JH UW S', 'socks S AA K S', 'shoes SH UW Z', 'different D IH F ER AH N T', 'languages L AE NG G W AH JH AH Z', 'system S IH S T AH M', 'open OW P AH N', 'settings S EH T IH NG Z', 'device D IH V AY S', 'ok OW K EY', 'now N AW', 'main M EY N', 'phones F OW N Z', 'practice P R AE K T IH S', 'light L AY T', 'cannot K AE N AA T', 'hi HH AY', 'morning M AO R N IH NG', 'evening IY V N IH NG', 'afternoon AE F T ER N UW N', 'goodbye G UH D B AY', 'bye B AY', 'see S IY', 'soon S UW N', 'later L EY T ER', 'can K AE N', 'tired T AY ER D', 'nap N AE P', 'want W AA N T', 'lie L AY', 'headache HH EH D EY K', 'head HH EH D', 'hurts HH ER T S', 'hungry HH AH NG G R IY', 'thirsty TH ER S T IY', 'backache B AE K EY K', 'back B AE K', 'aches EY K S', 'watch W AA CH', 'feel F IY L', 'computer K AH M P Y UW T ER', 'very V EH R IY', 'lot L AA T', 'exhausted IH G Z AO S T IH D', 'having HH AE V IH NG', 'getting G EH T IH NG', 'bad B AE D', 'hurting HH ER T IH NG', 'pains P EY N Z', 'adventure AE D V EH N CH ER', 'actual AE K CH AH W AH L', 'algorithm AE L G ER IH DH AH M', 'although AO L DH OW', 'another AH N AH DH ER', 'budget B AH JH IH T', 'challenge CH AE L AH N JH', 'fellowship F EH L OW SH IH P', 'fertilization F ER T AH L IH Z EY SH AH N', 'fiction F IH K SH AH N', 'garnish G AA R N IH SH', 'girlish G ER L IH SH', 'good G UH D', 'hook HH UH K', 'plural P L UH R AH L', 'premature P R IY M AH CH UH R', 'procure P R OW K Y UH R', 'wavelength W EY V L EH NG TH', 'wealth W EH L TH', 'with W IH TH']
phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'UH', 'UW', 'CH', 'D', 'DH', 'G', 'HH', 'JH', 'K', 'L', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'Y', 'Z', 'P', 'B', 'F', 'M', 'V', 'W']
phoneme_label_map = []

for word in word_map[:-1]:
	map_ = []
	ph_list = word.split()[1:]
	# print(ph_list)
	for ph in ph_list:
		map_.append(phonemes.index(ph))
	phoneme_label_map.append(map_)

print(phoneme_label_map)



