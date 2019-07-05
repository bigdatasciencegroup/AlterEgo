import os

data_dir = 'data/data'
files = map(lambda x: data_dir + '/' + x, filter(lambda x: '.txt' in x, os.listdir(data_dir)))

print files
print len(files)

total_time = 0.0
for file in files:
    lines = [line.strip() for line in open(file, 'rb').readlines()]
    sample_rate = float(lines[2].split()[3])
    num_samples = len(lines[6:])
    time = num_samples / sample_rate
    print time
    total_time += time
    
print
print total_time, 'seconds'
print total_time / 3600, 'hours'
