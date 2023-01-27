import tensorflow as tf
import numpy as np


def poisson_gen(x, nt, dtype=tf.float32):
    batch_size, features = x.shape
    res = tf.random.uniform((batch_size, features, nt))
    br_x = tf.broadcast_to(tf.expand_dims(x, -1), res.shape)
    return tf.cast(tf.where(res < br_x, 1, 0), dtype)

def batch_data(data, batch_size):
    data_size, n_features = data.shape
    n_batches = int(data_size / batch_size)
    batched_data = np.zeros((n_batches, batch_size, n_features))

    for i in range(n_batches):
        start = batch_size * i
        batched_data[i] = data[start: start + batch_size]
    return batched_data

def derivative_relu(x, alpha=0.1):
    return tf.where(x < 0, alpha, 1.0)