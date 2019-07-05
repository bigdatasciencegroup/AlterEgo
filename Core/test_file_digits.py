import numpy as np

import data

def on_data(history):
    print '\t'.join(map(str, history[-1]))

data.from_file.start('data/data/13_subvocal_3_50_trials.txt', on_data, speed=3.0)