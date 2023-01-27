from abc import ABC
from collections import namedtuple

import tensorflow as tf
from tensorflow import keras

# (membrane potential, 'resting times', 'drive (weighted input)')
InternalState = namedtuple('InternalState', ('u', 'r', 'v'))


class LIF(ABC, keras.layers.Layer):
    def __init__(self, n_neurons: int, n_err: int, rest: float = 0.0, threshold: float = 0.4, refractory: float = 1.0,
                 tau: float = 20.0, dt: float = 0.25, epsilon: float = 0.001,
                 train_fn_name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_neurons = n_neurons
        self.resting_potential = tf.cast(rest, self.dtype)
        self.threshold = tf.cast(threshold, self.dtype)
        self.refractory = tf.cast(refractory, self.dtype)
        self.tau = tf.cast(tau, self.dtype)
        self.eps = tf.cast(epsilon, self.dtype)
        self.dt = tf.cast(dt, self.dtype)

        self.W = None
        self.b = None
        if train_fn_name == 'ba':
            self.B = self.add_weight('B', shape=(n_err, n_neurons), initializer='GlorotUniform', trainable=False)
        self.state = None
        self.inputs = None
        self.alpha = tf.exp(-self.dt / self.tau)

    def build(self, input_shape: list) -> None:
        batch_size, n_inputs = input_shape
        self.inputs = tf.zeros(n_inputs, dtype=self.dtype)

        self.W = self.add_weight('W', shape=(n_inputs, self.n_neurons), initializer='GlorotUniform', trainable=True)
        self.b = self.add_weight('b', shape=(self.n_neurons,), initializer='GlorotUniform', trainable=True)
        self.state = self.zero_state(batch_size)

    def zero_state(self, batch_size: int) -> InternalState:
        return InternalState(
            u=tf.fill((batch_size, self.n_neurons), tf.cast(self.resting_potential, dtype=self.dtype)),
            r=tf.zeros((batch_size, self.n_neurons), self.dtype),
            v=tf.zeros((batch_size, self.n_neurons), self.dtype)
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        simulate one time-step, updates internal values and returns whether a spike occurred or not
        :param inputs: one dimensional tensor of inputs with size n_in
        :return: boolean tensor of spike or non-spike values
        """

        # old internal state
        u, r, _ = self.state

        v = inputs @ self.W + self.b
        new_r = tf.where(r > 0., r + self.dt, r)
        new_r = tf.where(new_r > self.refractory, 0., new_r)
        new_u = tf.where(new_r > 0., self.resting_potential, u + (v - u) * (self.dt / self.tau))
        new_r = tf.where(new_u > self.threshold, self.eps, new_r)

        # new observable state
        z = tf.cast(tf.where(new_r > 0, 1.0, 0.0), self.dtype)

        # new internal state
        self.state = InternalState(new_u, new_r, v)
        self.inputs = inputs

        return z

    def reset(self, batch_size: int):
        self.state = self.zero_state(batch_size)