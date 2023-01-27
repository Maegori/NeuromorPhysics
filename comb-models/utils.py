import tensorflow as tf


def poisson_gen(x, nt, dtype=tf.float32):
    """generate spike train for input x over nt timesteps as a poisson spike train"""
    batch_size, features = x.shape
    res = tf.random.uniform((batch_size, features, nt))
    br_x = tf.broadcast_to(tf.expand_dims(x, -1), res.shape)
    return tf.cast(tf.where(res < br_x, 1, 0), dtype)