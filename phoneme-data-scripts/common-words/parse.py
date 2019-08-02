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
 for line in open('common_words_phonemes.txt', 'r', encoding='ISO-8859-1')]

data = []

# with open('wordmap.txt', 'r') as file:
# 	filedata = file.read()

# filedata = filedata.replace('\n', '\', \'')

# open('wordmap.txt', 'w+').close() # To erase all previous contents of the file
# with open('wordmap.txt', 'w') as file:
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

# Distribution in entire generated dataset
# AA : 15   AE : 21   AH : 24   AO : 7   AW : 6   AY : 12   EH : 15   ER : 19   EY : 14   IH : 33   IY : 25   OW : 11  OY : 0  UH : 5   UW : 21
# CH : 8   D : 23  DH : 4   G : 10   HH : 15   JH : 5   K : 28   L : 25   N : 34   NG : 17   R : 22   S : 36   SH : 3   T : 49   TH : 6   Y : 6   Z : 14  ZH : 0  P : 18  B : 18  M : 23  F: 9  V : 10  W: 9
# Total Phonemes : 612

word_map = ['aches EY K S', 'afternoon AE F T ER N UW N', 'allow AH L AW', 'am AE M', 'anything EH N IY TH IH NG', 'audio AA D IY OW', 'back B AE K', 'backache B AE K EY K', 'bad B AE D', 'bathroom B AE TH R UW M', 'bed B EH D', 'bedroom B EH D R UW M', 'bring B R IH NG', 'bye B AY', 'can K AE N', 'cant K AE N T', 'cannot K AE N AA T', 'change CH EY N JH', 'chinese CH AY N IY Z', 'computer K AH M P Y UW T ER', 'cooler K UW L ER', 'could K UH D', 'couldnt K UH D AH N T', 'decrease D IH K R IY S', 'device D IH V AY S', 'different D IH F ER AH N T', 'down D AW N', 'drink D R IH NG K', 'eat IY T', 'english IH NG G L IH SH', 'evening IY V N IH NG', 'exhausted IH G Z AO S T IH D', 'far F AA R', 'feel F IY L', 'fetch F EH CH', 'german JH ER M AH N', 'get G EH T', 'getting G EH T IH NG', 'go G OW', 'good G UH D', 'goodbye G UH D B AY', 'have HH AE V', 'having HH AE V IH NG', 'head HH EH D', 'headache HH EH D EY K', 'hear HH IY R', 'heat HH IY T', 'heating HH IY T IH NG', 'hello HH AH L OW', 'help HH EH L P', 'hey HH EY', 'hi HH AY', 'hotter HH AA T ER', 'hungry HH AH NG G R IY', 'hurting HH ER T IH NG', 'hurts HH ER T S', 'in IH N', 'increase IH N K R IY S', 'into IH N T UW', 'is IH Z', 'it IH T', 'its IH T S', 'juice JH UW S', 'kitchen K IH CH AH N', 'korean K AO R IY AH N', 'lamp L AE M P', 'language L AE NG G W AH JH', 'languages L AE NG G W AH JH AH Z', 'later L EY T ER', 'less L EH S', 'levels L EH V AH L Z', 'lie L AY', 'light L AY T', 'lights L AY T S', 'lot L AA T', 'loud L AW D', 'louder L AW D ER', 'low L OW', 'lower L OW ER', 'main M EY N', 'make M EY K', 'max M AE K S', 'me M IY', 'more M AO R', 'morning M AO R N IH NG', 'much M AH CH', 'music M Y UW Z IH K', 'mute M Y UW T', 'my M AY', 'nap N AE P', 'need N IY D', 'newspaper N UW Z P EY P ER', 'now N AW', 'of AH V', 'off AO F', 'ok OW K EY', 'on AA N', 'open OW P AH N', 'pain P EY N', 'pains P EY N Z', 'pause P AO Z', 'phone F OW N', 'phones F OW N Z', 'play P L EY', 'please P L IY Z', 'practice P R AE K T IH S', 'put P UH T', 'quiet K W AY AH T', 'quieter K W AY AH T ER', 'reboot R IY B UW T', 'reduce R IH D UW S', 'restart R IY S T AA R T', 'resume R IH Z UW M', 'see S IY', 'set S EH T', 'settings S EH T IH NG Z', 'shoes SH UW Z', 'sleep S L IY P', 'so S OW', 'socks S AA K S', 'softer S AA F T ER', 'some S AH M', 'something S AH M TH IH NG', 'soon S UW N', 'sound S AW N D', 'start S T AA R T', 'stop S T AA P', 'switch S W IH CH', 'system S IH S T AH M', 'take T EY K', 'temperature T EH M P R AH CH ER', 'thank TH AE NG K', 'thanks TH AE NG K S', 'that DH AE T', 'thats DH AE T S', 'the DH AH', 'thirsty TH ER S T IY', 'this DH IH S', 'tired T AY ER D', 'to T UW', 'too T UW', 'turn T ER N', 'tv T IY V IY', 'up AH P', 'use Y UW S', 'very V EH R IY', 'video V IH D IY OW', 'volume V AA L Y UW M', 'want W AA N T', 'washroom W AA SH R UW M', 'watch W AA CH', 'water W AO T ER', 'you Y UW', 'FINISHED FINISHED FINISHED FINISHED FINISHED']
phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'CH', 'D', 'DH', 'G', 'HH', 'JH', 'K', 'L', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'Y', 'Z', 'ZH', 'P', 'B', 'F', 'M', 'V', 'W']
phoneme_label_map = []

for word in word_map[:-1]:
	map_ = []
	ph_list = word.split()[1:]
	# print(ph_list)
	for ph in ph_list:
		map_.append(phonemes.index(ph))
	phoneme_label_map.append(map_)

print(phoneme_label_map)



