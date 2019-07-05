import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import cv2

accuracy_map = {}
for filename in glob.glob('contribution/*.txt'):
    try:
        accuracy = float(open(filename, 'r').readlines()[-2].strip())
        print (filename[26:-4], accuracy)
        accuracy_map[filename[26:-4]] = accuracy
    except:
        pass
    
def get_children(node):
    if len(node) <= 1: return []
    return sum([zip(map(lambda x: ''.join(x), list(itertools.combinations(node, k))),
               map(lambda x: ''.join(x), list(itertools.combinations(node, len(node)-k)))[::-1]) for k in range(1, 2) if len(node) > k], [])
    
print
    
queue = ['01234567']
results = {}
visited = set()
while len(queue):
    node = queue.pop()
    if node in visited: continue
    visited.add(node)
    children = get_children(node)
    for channels, child in children:
        if child in accuracy_map:
#            accuracy_diff = accuracy_map[node] - accuracy_map[child]
#            accuracy_diff = accuracy_map[node]
            accuracy_diff = (accuracy_map[node] - accuracy_map[child]) / (1 - accuracy_map[child])
            print child, '=>', node, '\t', accuracy_diff
            if len(child) <= 3:
                results[channels] = results.get(channels, []) + [(accuracy_diff, child, node)]
            queue.append(child)

keys, results = zip(*results.items())
results = list(results)

print

print keys
print results
            
print

for i in range(len(results)):
    results[i] = list(sorted(results[i]))[::-1]
    
results2 = [0.0] * 8
    
avg_channel_contributions = []
for i in range(len(results)):
    avg = np.mean(map(float, np.array(results[i])[:,0]))
    avg_channel_contributions.append((avg, keys[i]))
    for j in range(len(keys[i])):
        results2[int(keys[i][j])] += avg
avg_channel_contributions = list(sorted(avg_channel_contributions))[::-1]
for i in range(len(avg_channel_contributions)):
    print avg_channel_contributions[i][1], avg_channel_contributions[i][0]

print

print results2
contributions = []
total_contributions = []
for i in range(len(results2)):
    total_contributions.append((results2[i], i))
    contributions.append(results2[i])
total_contributions = list(sorted(total_contributions))[::-1]
for i in range(len(total_contributions)):
    print total_contributions[i][1], total_contributions[i][0]

print

#test_index = 7
#print keys[test_index]
#for i in range(len(results[test_index])):
#    print results[test_index][i]


tmp = contributions[6]
contributions[6] = contributions[5]
contributions[5] = tmp

print contributions


im = cv2.imread('head_model2_resized.png')[:,:,::-1]

coords = [(314, 264), (491, 238), (514, 276), (424, 290), (369, 417), (425, 393), (486, 395), (528, 416)]
#contributions = [0.021206611838224049, 0.015588613674112375, 0.012038567763487383, 0.0043728192920970896, 0.0001395769574639991, 0.059316805905432313, 0.0088062444118518837, 0.0042626261656376939]
colors = [[121, 121, 121], [148, 67, 251], [16, 63, 251], [19, 141, 21], [254, 249, 55], [255, 147, 38], [255, 42, 28], [146, 82, 17]]

sqrts = map(np.sqrt, contributions)
for i in range(len(im)):
    for j in range(len(im[0])):
        for k in range(len(coords)):
            if np.linalg.norm(np.array([j, i]) - np.array(list(coords[k]))) <= (30.0 / max(sqrts)) * sqrts[k]:
                im[i,j,:] = colors[k]
                
cv2.imwrite('figure_electrode_contribution.png', im[:,:,::-1])

plt.figure(figsize=(12, 8))
plt.imshow(im)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plt.show()