import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

im = cv2.imread('head_model2_resized.png')
b,g,r = cv2.split(im)
im = cv2.merge([r,g,b])

coords = [(314, 264), (491, 238), (514, 276), (424, 290), (369, 417), (425, 393), (486, 395), (528, 416)]
contributions = [0.021206611838224049, 0.015588613674112375, 0.012038567763487383, 0.0043728192920970896, 0.0001395769574639991, 0.059316805905432313, 0.0088062444118518837, 0.0042626261656376939]
colors = [[121, 121, 121], [148, 67, 251], [16, 63, 251], [19, 141, 21], [254, 249, 55], [255, 147, 38], [255, 42, 28], [146, 82, 17]]

sqrts = map(np.sqrt, contributions)
for i in range(len(im)):
    for j in range(len(im[0])):
        for k in range(len(coords)):
            if np.linalg.norm(np.array([j, i]) - np.array(list(coords[k]))) <= (30.0 / max(sqrts)) * sqrts[k]:
                im[i,j,:] = colors[k]

plt.figure(figsize=(12, 8))
plt.imshow(im)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plt.show()