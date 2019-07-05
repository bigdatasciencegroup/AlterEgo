import itertools
import os

def run_trials(channels):
    filename = 'contribution_' + ''.join(map(str, channels)) + '.txt'
    exists = os.path.isfile(filename)
    if not exists:
        print ''.join(map(str, channels))
#        os.system('python final_digits_train_args.py "[' + ','.join(map(str, channels)) + ']" | tee ' + filename)
    else:
        print 'Skipping'

channel_sets = []
for i in range(8, 2, -1):
    channel_sets += list(itertools.combinations(range(8), i))
for i in range(len(channel_sets)):
    os.system('date')
    print str(i+1) + '/' + str(len(channel_sets)), channel_sets[i]
    run_trials(channel_sets[i])
    print

