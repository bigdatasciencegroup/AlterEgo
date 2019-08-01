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

with open('wordmap.txt', 'r') as file:
	filedata = file.read()

filedata = filedata.replace('\n', '\', \'')

open('wordmap.txt', 'w+').close() # To erase all previous contents of the file
with open('wordmap.txt', 'w') as file:
	file.write(filedata)


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


