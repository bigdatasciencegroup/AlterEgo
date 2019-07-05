##### Creating the dataset by pruning cmudict-0_7b


import numpy as np
from nltk.util import ngrams
from pandas import DataFrame

lines = [line.rstrip('\n')\
 for line in open('cmudict-0_7b.txt', 'r', encoding='ISO-8859-1')]

data = []
new_data = []
grams_list = []

for line in lines:	data.append(line.split())
data = np.asarray(data)

# print(data)

AA = 0
AE = 0
AH = 0
AO = 0
AW = 0
AY = 0
EH = 0
ER = 0
EY = 0
IH = 0
IY = 0
OW = 0
OY = 0
UH = 0
UW = 0

CH = 0
D = 0
DH = 0
G = 0
HH = 0
JH = 0
K = 0
L = 0
N = 0
NG = 0
R = 0
S = 0
SH = 0
T = 0
TH = 0
Y = 0
Z = 0
ZH = 0

for index,rows in enumerate(data):
	if 'P' in rows:		pass
	elif 'B' in rows:	pass
	elif 'F' in rows:	pass
	elif 'M' in rows:	pass
	elif 'V' in rows:	pass
	elif 'W' in rows:	pass
	else:
		rows = [element.replace('AA0','AA') for element in rows]
		rows = [element.replace('AA1','AA') for element in rows]
		rows = [element.replace('AA2','AA') for element in rows]

		rows = [element.replace('AE0','AE') for element in rows]
		rows = [element.replace('AE1','AE') for element in rows]
		rows = [element.replace('AE2','AE') for element in rows]
		
		rows = [element.replace('AH0','AH') for element in rows]
		rows = [element.replace('AH1','AH') for element in rows]
		rows = [element.replace('AH2','AH') for element in rows]
		
		rows = [element.replace('AO0','AO') for element in rows]
		rows = [element.replace('AO1','AO') for element in rows]
		rows = [element.replace('AO2','AO') for element in rows]

		rows = [element.replace('AW0','AW') for element in rows]
		rows = [element.replace('AW1','AW') for element in rows]
		rows = [element.replace('AW2','AW') for element in rows]

		rows = [element.replace('AY0','AY') for element in rows]
		rows = [element.replace('AY1','AY') for element in rows]
		rows = [element.replace('AY2','AY') for element in rows]

		rows = [element.replace('EH0','EH') for element in rows]
		rows = [element.replace('EH1','EH') for element in rows]
		rows = [element.replace('EH2','EH') for element in rows]

		rows = [element.replace('ER0','ER') for element in rows]
		rows = [element.replace('ER1','ER') for element in rows]
		rows = [element.replace('ER2','ER') for element in rows]

		rows = [element.replace('EY0','EY') for element in rows]
		rows = [element.replace('EY1','EY') for element in rows]
		rows = [element.replace('EY2','EY') for element in rows]

		rows = [element.replace('IH0','IH') for element in rows]
		rows = [element.replace('IH1','IH') for element in rows]
		rows = [element.replace('IH2','IH') for element in rows]

		rows = [element.replace('IY0','IY') for element in rows]
		rows = [element.replace('IY1','IY') for element in rows]
		rows = [element.replace('IY2','IY') for element in rows]

		rows = [element.replace('OW0','OW') for element in rows]
		rows = [element.replace('OW1','OW') for element in rows]
		rows = [element.replace('OW2','OW') for element in rows]

		rows = [element.replace('OY0','OY') for element in rows]
		rows = [element.replace('OY1','OY') for element in rows]
		rows = [element.replace('OY2','OY') for element in rows]

		rows = [element.replace('UH0','UH') for element in rows]
		rows = [element.replace('UH1','UH') for element in rows]
		rows = [element.replace('UH2','UH') for element in rows]

		rows = [element.replace('UW0','UW') for element in rows]
		rows = [element.replace('UW1','UW') for element in rows]
		rows = [element.replace('UW2','UW') for element in rows]

		AA += rows.count('AA')
		AE += rows.count('AE')
		AH += rows.count('AH')
		AO += rows.count('AO')
		AW += rows.count('AW')
		AY += rows.count('AY')
		EH += rows.count('EH')
		ER += rows.count('ER')
		EY += rows.count('EY')
		IH += rows.count('IH')
		IY += rows.count('IY')
		OW += rows.count('OW')
		OY += rows.count('OY')
		UH += rows.count('UH')
		UW += rows.count('UW')

		CH += rows.count('CH')
		D += rows.count('D')
		DH += rows.count('DH')
		G += rows.count('G')
		HH += rows.count('HH')
		JH += rows.count('JH')
		K += rows.count('K')
		L += rows.count('L')
		N += rows.count('N')
		NG += rows.count('NG')
		R += rows.count('R')
		S += rows.count('S')
		SH += rows.count('SH')
		T += rows.count('T')
		TH += rows.count('TH')
		Y += rows.count('Y')
		Z += rows.count('Z')
		ZH += rows.count('ZH')

		new_data.append(rows)

		tokens = rows[1:]
		grams = list(ngrams(tokens, 3))
		grams_list.append(grams)

# print(data.shape)
new_data = np.asarray(new_data)
# print(new_data.shape)
# print(new_data)
print('AA :',AA, '  AE :',AE, '  AH :',AH, '  AO :',AO, '  AW :',AW, '  AY :',AY, '  EH :',EH, '  ER :',ER, '  EY :',EY, '  IH :',IH, '  IY :',IY, '  OW :',OW, '  OY :',OY, '  UH :',UH, '  UW :',UW)
print('CH :',CH, '  D :',D, '  DH :',DH, '  G :',G, '  HH :',HH, '  JH :',JH, '  K :',K, '  L :',L, '  N :',N, '  NG :',NG, '  R :',R, '  S :',S, '  SH :',SH, '  T :',T, '  TH :',TH, '  Y :',Y, '  Z :',Z, '  ZH :',ZH)

df = DataFrame.from_records(grams_list)
# print(df)
# print('DF SIZE', df.size)

df.drop_duplicates(subset=0, inplace=True)
# print(df)
# print('MODIFIED DF SIZE', df.size)

data = list(df[0])
# print(len(data))
data.remove(None)
# print(data)
print('Total words:',len(data))

AA = 0
AE = 0
AH = 0
AO = 0
AW = 0
AY = 0
EH = 0
ER = 0
EY = 0
IH = 0
IY = 0
OW = 0
OY = 0
UH = 0
UW = 0

CH = 0
D = 0
DH = 0
G = 0
HH = 0
JH = 0
K = 0
L = 0
N = 0
NG = 0
R = 0
S = 0
SH = 0
T = 0
TH = 0
Y = 0
Z = 0
ZH = 0

for index,rows in enumerate(data):
	AA += rows.count('AA')
	AE += rows.count('AE')
	AH += rows.count('AH')
	AO += rows.count('AO')
	AW += rows.count('AW')
	AY += rows.count('AY')
	EH += rows.count('EH')
	ER += rows.count('ER')
	EY += rows.count('EY')
	IH += rows.count('IH')
	IY += rows.count('IY')
	OW += rows.count('OW')
	OY += rows.count('OY')
	UH += rows.count('UH')
	UW += rows.count('UW')

	CH += rows.count('CH')
	D += rows.count('D')
	DH += rows.count('DH')
	G += rows.count('G')
	HH += rows.count('HH')
	JH += rows.count('JH')
	K += rows.count('K')
	L += rows.count('L')
	N += rows.count('N')
	NG += rows.count('NG')
	R += rows.count('R')
	S += rows.count('S')
	SH += rows.count('SH')
	T += rows.count('T')
	TH += rows.count('TH')
	Y += rows.count('Y')
	Z += rows.count('Z')
	ZH += rows.count('ZH')

print('AA :',AA, '  AE :',AE, '  AH :',AH, '  AO :',AO, '  AW :',AW, '  AY :',AY, '  EH :',EH, '  ER :',ER, '  EY :',EY, '  IH :',IH, '  IY :',IY, '  OW :',OW, '  OY :',OY, '  UH :',UH, '  UW :',UW)
print('CH :',CH, '  D :',D, '  DH :',DH, '  G :',G, '  HH :',HH, '  JH :',JH, '  K :',K, '  L :',L, '  N :',N, '  NG :',NG, '  R :',R, '  S :',S, '  SH :',SH, '  T :',T, '  TH :',TH, '  Y :',Y, '  Z :',Z, '  ZH :',ZH)



