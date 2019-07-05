import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

print(matplotlib.get_backend())

plt.plot([1,2,3,4])
plt.show()