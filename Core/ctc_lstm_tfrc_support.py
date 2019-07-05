
# ...
# ...
# ...

inputs = tf.placeholder(tf.float32,[None, None, num_channels]) #[batch_size, timestep, features]
targets = tf.placeholder(tf.int32, [None, None])
sequence_lengths = tf.placeholder(tf.int32, [None])
label_lengths = tf.placeholder(tf.int32, [None])
weights = tf.placeholder(tf.float32, [None])
training = tf.placeholder(tf.bool)

model_inputs = [
    inputs,
    targets,
    sequence_lengths,
    label_lengths,
    weights,
    training,
]

def outside_fn(logits, loss, targets, sequence_lengths):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='optimizer')
    
    idx = tf.where(tf.not_equal(targets, 0))
    targets_sparse = tf.SparseTensor(idx, tf.gather_nd(targets, idx), tf.cast(tf.shape(targets), tf.int64))

    ctc_output = tf.nn.ctc_beam_search_decoder(logits, sequence_lengths)
    decoded, log_prob = ctc_output[0][0], ctc_output[1][0]

    error = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), targets_sparse, normalize=True))
    
    decoded = tf.sparse_tensor_to_dense(decoded, default_value=-1)

def model_computation(inputs, targets, sequence_lengths, label_lengths, weights, training):
    batch_size = tf.shape(inputs)[0]

    # Two layer bidirectional LSTM with dropout
    lstm_hidden_size = 256
    lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
    dropout1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, 1.0-dropout_rate)
    lstm1b = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
    dropout1b = tf.nn.rnn_cell.DropoutWrapper(lstm1b, 1.0-dropout_rate)
    lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
    dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, 1.0-dropout_rate)
    lstm2b = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, use_peepholes=True)
    dropout2b = tf.nn.rnn_cell.DropoutWrapper(lstm2b, 1.0-dropout_rate)

    forward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout2])
    backward_stack = tf.nn.rnn_cell.MultiRNNCell([dropout2b])

    outputs, states = tf.nn.bidirectional_dynamic_rnn(forward_stack, backward_stack, inputs, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)

    # Time-distributed dense layers and output
    reshaped = tf.reshape(outputs, [-1, 2 * lstm_hidden_size])
    dense1 = tf.layers.dense(reshaped, 1024, activation=tf.nn.relu,
                kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(2 * lstm_hidden_size * 1024))))
    dense2 = tf.layers.dense(dense1, 1024, activation=tf.nn.relu,
                kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(1024 * 1024))))
    logits = tf.layers.dense(dense2, num_classes,
                kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2.0/(1024 * num_classes))))

    # CTC loss
    logits = tf.reshape(logits, [batch_size, -1, num_classes])
    logits = tf.transpose(logits, [1, 0, 2])

    loss = tf.nn.ctc_loss_v2(labels=targets, logits=logits, label_length=label_lengths, logit_length=sequence_lengths,
                             logits_time_major=True)
    loss = tf.reduce_mean(tf.multiply(loss, weights))

    # Outside compilation to compute training op, and prediction output and error metrics (uses sparse tensor)
    tpu.outside_compilation(outside_fn, logits, loss, targets, sequence_lengths)
    
    return logits, loss

logits, loss = tpu.rewrite(model_computation, model_inputs)

# Get train op variable
optimizer = tf.get_default_graph().get_operation_by_name('optimizer')

# ...
# ...
# ...
