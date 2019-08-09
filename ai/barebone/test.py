import tensorflow as tf
import numpy as np

device = '/cpu:0'
learning_rate = 1e-2
input_shape = (32, 100)
hidden_shape = 1000
output_shape = 10

tf.reset_default_graph()

with tf.device(device):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.int32)
    w1 = tf.Variable(tf.random_normal((input_shape[1], hidden_shape)))
    w2 = tf.Variable(tf.random_normal((hidden_shape, output_shape)))

    h = tf.nn.relu(tf.matmul(x, w1))
    scores = tf.matmul(h, w2)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(losses)

    params = [w1, w2]
    grad_params = tf.gradients(loss, params)

    new_weights = []
    for w, grad_w in zip(params, grad_params):
        new_w = tf.assign_sub(w, learning_rate * grad_w)
        new_weights.append(new_w)
        print(new_w)

    with tf.control_dependencies(new_weights):
        loss_ret = tf.identity(loss)

x_np = np.zeros(input_shape)
y_np = np.zeros(input_shape[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_np = sess.run(loss_ret, feed_dict={x: x_np, y: y_np})
    print(loss_np)
