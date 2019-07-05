import data

def on_data(history):
    print '\t'.join(map(str, history[-1]))
    
# 161_012_30_trials.txt
# 168_012_1k_30.txt    
data.from_file.start('data/data/168_012_1k_30.txt', on_data, sample_rate=1000,
                     transform_fn=data.transform.default_transform_new, default_step=10, starting_point=60000)

