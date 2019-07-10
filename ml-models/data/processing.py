import numpy as np


def process(num_classes, filepaths, include_surrounding=True, sample_rate=250, channels=range(0, 8), surrounding=150):
    def get_sequence_groups(filepath):
        print 'Processing', filepath
        f = open(filepath, 'r')
        contents = map(lambda x: x.strip(), f.readlines())
        frames_original = filter(lambda x: x and x[0] != '%', contents)[1:]
        frames_original = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames_original)

        # assert equal data vector lengths
        assert len(set(map(len, frames_original))) == 1, map(len, frames_original)

        # (7 channels + digital trigger)
        frames = map(lambda a: map(float, a[1:9]) + [float(a[12] if sample_rate==250 else a[11])], frames_original)
        timestamps = map(lambda a: a[-1], frames_original)
        frames = np.array(frames)

        speaking = False
        start_index = 0
        num = -1
        sequence_groups = [[] for _ in range(num_classes)]
        padding = surrounding*sample_rate//250 if include_surrounding else 0
        for i in range(len(frames)):
            if not speaking and bool(frames[i][-1]):
                speaking = True
                start_index = i
            if speaking and not bool(frames[i][-1]):
                speaking = False
                if np.all(map(bool, frames[i-50:i,-1])):
                    num += 1
#                    print i - start_index
                    sequence_groups[num % num_classes].append(frames[start_index-padding:i+padding,channels])
#                    print num, timestamps[start_index]
        if bool(frames[-1][-1]):
            sequence_groups[num % num_classes].append(frames[start_index-padding:,channels])

        # assert equal number of sequences for each class
        assert len(set(map(len, sequence_groups))) == 1, map(len, sequence_groups)

        return np.array(sequence_groups)

    def join_sequence_groups(*g):
        assert len(g), len(g)
        assert len(set(map(len, g))) == 1, map(len, g)
        groups = [[] for _ in range(len(g[0]))]
        for i in range(len(g)):
            for j in range(len(g[0])):
                groups[j] += list(g[i][j])
        return np.array(groups)
    
    sequence_groups = map(get_sequence_groups, filepaths)
    return join_sequence_groups(*sequence_groups)

def process_scrambled(labels, filepaths, include_surrounding=True, sample_rate=250, channels=range(0, 8), surrounding=150, exclude=set(),
                      num_classes=None):
    def get_sequence_groups(filepath):
        print 'Processing', filepath
        f = open(filepath, 'r')
        contents = map(lambda x: x.strip(), f.readlines())
        frames_original = filter(lambda x: x and x[0] != '%', contents)[1:]
        frames_original = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames_original)

        # assert equal data vector lengths
#        assert len(set(map(len, frames_original))) == 1, map(len, frames_original)

        # (7 channels + digital trigger)
        frames = map(lambda a: map(float, a[1:9]) + [float(a[12] if sample_rate==250 else a[11])], frames_original)
        timestamps = map(lambda a: a[-1], frames_original)
        frames = np.array(frames)

        speaking = False
        start_index = 0
        num = -1
#       sequence_groups = [[] for _ in range(len(np.unique(labels)))]
        sequence_groups = [[] for _ in range(max(labels)+1 if num_classes is None else num_classes)]
        padding = surrounding*sample_rate//250 if include_surrounding else 0
        for i in range(len(frames)):
            if not speaking and bool(frames[i][-1]):
                speaking = True
                start_index = i
            if speaking and not bool(frames[i][-1]):
                speaking = False
                if np.all(map(bool, frames[i-50:i,-1])):
                    num += 1
                    if num not in exclude:
                        sequence_groups[labels[num]].append(frames[start_index-padding:i+padding, channels])
        if bool(frames[-1][-1]):
            sequence_groups[labels[num]].append(frames[start_index-padding:,channels])

        # assert equal number of sequences for each class
#        assert len(set(map(len, sequence_groups))) == 1, map(len, sequence_groups)

        return np.array(sequence_groups)

    def join_sequence_groups(*g):
        assert len(g), len(g)
        assert len(set(map(len, g))) == 1, map(len, g)
        groups = [[] for _ in range(len(g[0]))]
        for i in range(len(g)):
            for j in range(len(g[0])):
                groups[j] += list(g[i][j])
        return np.array(groups)
    
    sequence_groups = map(get_sequence_groups, filepaths)
    return join_sequence_groups(*sequence_groups)


def process_silence_between(filepaths, include_surrounding=True, sample_rate=250, channels=range(0, 8), surrounding=-150):
    num_classes = 1
    def get_sequence_groups(filepath):
        print 'Processing', filepath
        f = open(filepath, 'r')
        contents = map(lambda x: x.strip(), f.readlines())
        frames_original = filter(lambda x: x and x[0] != '%', contents)[1:]
        frames_original = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames_original)

        # assert equal data vector lengths
        assert len(set(map(len, frames_original))) == 1, map(len, frames_original)

        # (7 channels + digital trigger)
        frames = map(lambda a: map(float, a[1:9]) + [float(a[12] if sample_rate==250 else a[11])], frames_original)
        timestamps = map(lambda a: a[-1], frames_original)
        frames = np.array(frames)
        frames[:,-1] = 1 - frames[:,-1]

        speaking = False
        start_index = 0
        num = -1
        sequence_groups = [[] for _ in range(num_classes)]
        padding = surrounding*sample_rate//250 if include_surrounding else 0
        for i in range(len(frames)):
            if not speaking and bool(frames[i][-1]):
                speaking = True
                start_index = i
            if speaking and not bool(frames[i][-1]):
                speaking = False
                if np.all(map(bool, frames[i-50:i,-1])):
                    num += 1
#                    print i - start_index
                    if 250 < i - start_index < 1500:
                        sequence_groups[num % num_classes].append(frames[start_index-padding:i+padding,channels])
#                    print num, timestamps[start_index]
        if bool(frames[-1][-1]):
            sequence_groups[num % num_classes].append(frames[start_index-padding:,channels])
            
        print min(map(len, sequence_groups[0]))

        # assert equal number of sequences for each class
        assert len(set(map(len, sequence_groups))) == 1, map(len, sequence_groups)

        return np.array(sequence_groups)

    def join_sequence_groups(*g):
        assert len(g), len(g)
        assert len(set(map(len, g))) == 1, map(len, g)
        groups = [[] for _ in range(len(g[0]))]
        for i in range(len(g)):
            for j in range(len(g[0])):
                groups[j] += list(g[i][j])
        return np.array(groups)
    
    sequence_groups = map(get_sequence_groups, filepaths)
    return join_sequence_groups(*sequence_groups)

def process_scrambled_flattened(labels, filepaths, include_surrounding=True, sample_rate=250, channels=range(0, 8), surrounding=150):
    def get_sequences(filepath):
        print 'Processing', filepath
        f = open(filepath, 'r')
        contents = map(lambda x: x.strip(), f.readlines())
        frames_original = filter(lambda x: x and x[0] != '%', contents)[1:]
        frames_original = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames_original)

        # assert equal data vector lengths
#        assert len(set(map(len, frames_original))) == 1, map(len, frames_original)

        # (7 channels + digital trigger)
        frames = map(lambda a: map(float, a[1:9]) + [float(a[-3] if sample_rate==250 else a[-2])], frames_original)
        timestamps = map(lambda a: a[-1], frames_original)
        frames = np.array(frames)

        speaking = False
        start_index = 0
        num = -1
        sequences = []
        padding = surrounding*sample_rate//250 if include_surrounding else 0
        for i in range(len(frames)):
            if not speaking and bool(frames[i][-1]):
                speaking = True
                start_index = i
                num += 1
            if speaking and not bool(frames[i][-1]):
                speaking = False
                sequences.append(frames[start_index-padding:i+padding,channels])
#                print num, timestamps[start_index]
        if bool(frames[-1][-1]):
            sequences.append(frames[start_index-padding:,channels])

        return np.array(sequences)
    
    sequences = sum(map(list, map(get_sequences, filepaths)), [])
    return sequences

def process_file(filepath, sample_rate=250, channels=range(4, 8)):
    print 'Processing', filepath
    f = open(filepath, 'r')
    contents = map(lambda x: x.strip(), f.readlines())
    frames_original = filter(lambda x: x and x[0] != '%', contents)[1:]
    frames_original = map(lambda s: map(lambda ss: ss.strip(), s.split(',')), frames_original)

    # assert equal data vector lengths
    assert len(set(map(len, frames_original))) == 1, map(len, frames_original)

    # (7 channels + digital trigger)
    frames = map(lambda a: map(float, a[1:9]) + [float(a[-3] if sample_rate==250 else a[-2])], frames_original)
    timestamps = map(lambda a: a[-1], frames_original)
    frames = np.array(frames)
#    return np.concatenate([frames[:,channels], frames[:,[-1]]*100], axis=1)
    return frames[:,channels]

def split(sequence_groups, test_train_split, seed=0):
    # Fix random seed for consistent test/train split
    np.random.seed(seed)
    
    for i in range(len(sequence_groups)):
        np.random.shuffle(sequence_groups[i])
    
    # Split into training/validation sets
    validation_sequence_groups = [None] * len(sequence_groups)
    training_sequence_groups = [None] * len(sequence_groups)
    for i in range(len(sequence_groups)):
        validation_indices = np.random.choice(range(len(sequence_groups[i])),
                                              int(test_train_split * len(sequence_groups[i])),
                                              replace=False)
        validation_selection = np.in1d(range(len(sequence_groups[i])), validation_indices)
        validation_sequence_groups[i] = np.array(sequence_groups[i])[validation_selection]
        training_sequence_groups[i] = np.array(sequence_groups[i])[~validation_selection]
        
    return training_sequence_groups, validation_sequence_groups

def get_inputs(sequence_groups, seed=1):
    # Fix random seed for consistent ordering of data
    np.random.seed(seed)
        
    # Append label to each sequence, join into single list, and shuffle
    sequence_pairs = map(lambda (i, x): map(lambda y: (y, i), x), enumerate(sequence_groups))
    sequence_pairs = np.array(reduce(lambda a, b: a + b, sequence_pairs, []))
    np.random.shuffle(sequence_pairs)

    # Separate sequence and label
    sequences = sequence_pairs[:,0]
    labels = sequence_pairs[:,1]
    
    sequences = np.array(map(np.array, sequences))
    
    return sequences, labels