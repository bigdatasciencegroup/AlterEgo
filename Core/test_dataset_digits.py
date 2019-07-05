import numpy as np

import data

sequence_groups = data.digits_and_silence_dataset()
sequence_groups = data.transform.default_transform(sequence_groups)

print np.shape(sequence_groups)