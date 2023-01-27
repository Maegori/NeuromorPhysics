from abc import ABC
from grad_functions import surrogate_gradient_builder, broadcast_alignment_builder
from layer import LIF

import tensorflow as tf
from tensorflow import keras


class LIFNetwork(keras.Model, ABC):
    def __init__(self, sizes, nt, burn_in, grad_fn_name, surrogate_fn, gen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nt = nt
        self.burn_in = burn_in
        self.gen = gen
        out_size = None
        if grad_fn_name == 'ba':
            out_size = sizes[-1]
            self.grad_fn = broadcast_alignment_builder(surrogate_fn)
        else:
            self.grad_fn = surrogate_gradient_builder(surrogate_fn)

        self.ls = [LIF(size, out_size, train_fn_name=grad_fn_name, dtype=self.dtype) for size in sizes]

    def call(self, inputs, training=None, mask=None):
        out = inputs
        for layer in self.ls:
            out = layer(out)
        return out

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, self.dtype)
        x = self.gen(x, self.nt, self.dtype)

        for layer in self.ls:
            layer.reset(x.shape[0])
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
        x, y = data
        for layer in self.ls:
            layer.reset(x.shape[0])
        total_out = tf.zeros_like(y)
        for t in range(self.nt):
            out = self(x)

            if t > self.burn_in:
                # logging:
                total_out += out

        self.compiled_metrics.update_state(tf.nn.softmax(total_out, axis=1), y)
        return {m.name: m.result() for m in self.metrics}