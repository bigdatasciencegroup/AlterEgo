# DATA CONFIG

file_path = './data/data_maps.json'

channels = range(0, 8)
surrounding = 250
sample_rate = 250
# number of words in dataset
num_classes = 4757

# MODEL CONFIG
model = ""  # lstm, bisltm, bi-att

num_epochs = 200
batch_size = 80
learning_rate = 1e-2
decay = 0.1
latent_dim = 512

# EVAL CONFIG
num_folds = 5