from abc import ABC
from grad_functions import surrogate_gradient_builder, broadcast_alignment_builder
from layer import LIF

import tensorflow as tf
from tensorflow import keras


class LIFNetwork(keras.Model, ABC):
    def __init__(self, sizes, nt, burn_in, grad_fn_name, surrogate_fn, gen, *args, **kwargs):
        """
        network of lif layers
        :param sizes: list of layer sizes (excluding input layer)
        :param nt: number of simulation timesteps
        :param burn_in: first timesteps where no learning happens
        :param grad_fn_name: function name of the algorithm used
        :param surrogate_fn: surrogate function
        :param gen: input generator
        """
        super().__init__(*args, **kwargs)
        self.nt = nt
        self.burn_in = burn_in
        self.gen = gen
        out_size = None
        # set gradient function
        if grad_fn_name == 'ba':
            out_size = sizes[-1]
            self.grad_fn = broadcast_alignment_builder(surrogate_fn)
        else:
            self.grad_fn = surrogate_gradient_builder(surrogate_fn)
        # create LIF layers
        self.ls = [LIF(size, out_size, train_fn_name=grad_fn_name, dtype=self.dtype) for size in sizes]

    def call(self, inputs, training=None, mask=None):
        """ propagate input through the network """
        out = inputs
        for layer in self.ls:
            out = layer(out)
        return out

    def train_step(self, data):
        """ training function """
        x, y = data
        # cast input data to desired datatype and create input spike train
        y = tf.cast(y, self.dtype)
        x = self.gen(x, self.nt, self.dtype)

        # prepare the internal states for each layer
        for layer in self.ls:
            layer.reset(x.shape[0])

        # output spike train accumulator
        total_out = tf.zeros_like(y, dtype=self.dtype)

        for t in range(self.nt):
            out = self(x[:, :, t])
            if t > self.burn_in:
                # logging:
                total_out += out

                # updating:
                error = out - y
                grads = self.grad_fn(error, self.ls)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(tf.nn.softmax(total_out, axis=1), y)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """ evaluating function """
        x, y = data

        # cast input data to desired datatype and create input spike train
        y = tf.cast(y, self.dtype)
        x = self.gen(x, self.nt, self.dtype)

        # prepare the internal states for each layer
        for layer in self.ls:
            layer.reset(x.shape[0])

        # output spike train accumulator
        total_out = tf.zeros_like(y)
        for t in range(self.nt):
            out = self(x)

            if t > self.burn_in:
                # logging:
                total_out += out

        self.compiled_metrics.update_state(tf.nn.softmax(total_out, axis=1), y)
        return {m.name: m.result() for m in self.metrics}
