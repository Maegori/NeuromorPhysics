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
        """
        class for a single layer of LIF neurons
        :param n_neurons: number of neurons
        :param n_err: error signal dimension
        :param rest: resting period
        :param threshold: firing threshold
        :param refractory: refractory period
        :param tau: leak constant
        :param dt: dt
        :param epsilon: some small value that should be positive and smaller than rest
        :param train_fn_name: string name of the gradient function
        """
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
        # create random feedback weights for broadcast alignment
        if train_fn_name == 'ba':
            self.B = self.add_weight('B', shape=(n_err, n_neurons), initializer='GlorotUniform', trainable=False)
        self.state = None
        self.inputs = None
        self.alpha = tf.exp(-self.dt / self.tau)

    def build(self, input_shape: list) -> None:
        """build weights on first call of the layer"""
        batch_size, n_inputs = input_shape
        self.inputs = tf.zeros(n_inputs, dtype=self.dtype)

        self.W = self.add_weight('W', shape=(n_inputs, self.n_neurons), initializer='GlorotUniform', trainable=True)
        self.b = self.add_weight('b', shape=(self.n_neurons,), initializer='GlorotUniform', trainable=True)
        self.state = self.zero_state(batch_size)

    def zero_state(self, batch_size: int) -> InternalState:
        """returns the initial state with its initial values"""
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

        v = inputs @ self.W + self.b                    # compute drive
        new_r = tf.where(r > 0., r + self.dt, r)        # update resting period for already resting neurons
        new_r = tf.where(new_r > self.refractory, 0., new_r)  # complete resting period of rested neurons
        new_u = tf.where(new_r > 0., self.resting_potential, u + (v - u) * (self.dt / self.tau))  # compute new membrane potentials
        new_r = tf.where(new_u > self.threshold, self.eps, new_r)  # set neurons that fired to resting

        # new observable state
        z = tf.cast(tf.where(new_r > 0, 1.0, 0.0), self.dtype)

        # new internal state
        self.state = InternalState(new_u, new_r, v)
        self.inputs = inputs

        return z

    def reset(self, batch_size: int):
        """
        return the internal state with its init values
        """
        self.state = self.zero_state(batch_size)