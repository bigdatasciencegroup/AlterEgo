import numpy as np

training_sentences = list(range(75, 300))
test_separate_sentences = list(range(0, 75))


np.random.seed(1)

print '# timit300_train_1'
print '#labels =', list(np.random.choice(training_sentences, 200, replace=False))
print
print '# timit300_test1_1'
print '#labels =', list(np.random.choice(training_sentences, 50, replace=False))
print
print '# timit300_test2_1'
print '#labels =', list(np.random.choice(test_separate_sentences, 50, replace=False))
print
print '# timit300_train_2'
print '#labels =', list(np.random.choice(training_sentences, 200, replace=False))
print
print '# timit300_test1_2'
print '#labels =', list(np.random.choice(training_sentences, 50, replace=False))
print
print '# timit300_test2_2'
print '#labels =', list(np.random.choice(test_separate_sentences, 50, replace=False))
print
print '# timit300_train_3'
print '#labels =', list(np.random.choice(training_sentences, 200, replace=False))
print
print '# timit300_test1_3'
print '#labels =', list(np.random.choice(training_sentences, 50, replace=False))
print
print '# timit300_test2_3'
print '#labels =', list(np.random.choice(test_separate_sentences, 50, replace=False))
print
#print '# timit300_train_4'
#print '#labels =', list(np.random.choice(training_sentences, 200, replace=False))
#print
#print '# timit300_test1_4'
#print '#labels =', list(np.random.choice(training_sentences, 25, replace=False))
#print
#print '# timit300_test2_4'
#print '#labels =', list(np.random.choice(test_separate_sentences, 25, replace=False))
#print
#print '# timit300_train_5'
#print '#labels =', list(np.random.choice(training_sentences, 100, replace=False))
#print
#print '# timit300_test1_5'
#print '#labels =', list(np.random.choice(training_sentences, 25, replace=False))
#print
#print '# timit300_test2_5'
#print '#labels =', list(np.random.choice(test_separate_sentences, 25, replace=False))
#print
#print '# timit300_train_6'
#print '#labels =', list(np.random.choice(training_sentences, 100, replace=False))
#print
#print '# timit300_test1_6'
#print '#labels =', list(np.random.choice(training_sentences, 25, replace=False))
#print
#print '# timit300_test2_6'
#print '#labels =', list(np.random.choice(test_separate_sentences, 25, replace=False))
#print
#print '# timit300_train_7'
#print '#labels =', list(np.random.choice(training_sentences, 100, replace=False))
#print
