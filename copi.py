import tensorflow as tf
from tensorflow import keras

import rule_book


class CopiLayer(keras.layers.Layer):
    def __init__(self, units, activation, n_err=None, *args, **kwargs):
        super(CopiLayer, self).__init__(*args, **kwargs)
        self.n = units
        self.activation = activation
        if n_err is not None:
            self.B = self.add_weight(name='W', shape=(n_err, self.n),
                                     initializer='GlorotNormal',
                                     trainable=False)

        self.W = None
        self.R = None

    def build(self, input_shape):
        _, n_in = input_shape

        self.W = self.add_weight(name='W', shape=(n_in, self.n), initializer='GlorotNormal', trainable=True)
        self.R = self.add_weight(name='R', shape=(n_in, n_in), initializer='Identity', trainable=True)

    def call(self, y_prev, *args, **kwargs):
        x = y_prev @ self.R
        a = x @ self.W
        y = self.activation(a)
        return x, a, y


class CopiNetwork(keras.Model):
    def __init__(self, units, activations, lr_w, lr_r, df, alpha, rule='bp', *args, **kwargs):
        super(CopiNetwork, self).__init__(*args, **kwargs)
        n_err = None
        if rule == 'bp':
            self.rule = rule_book.copi_backpropagation
        elif rule == 'ba':
            self.rule = rule_book.copi_broadcast_alignment
            n_err = units[-1]
        else:
            print(f"rule not supported yet; choose between 'ba' and 'bp'")

        self.ls = [CopiLayer(u, a, n_err) for u, a in zip(units, activations)]
        self.df = df
        self.alpha = alpha
        self.lr_w = lr_w
        self.lr_r = lr_r
        if rule == 'bp':
            self.rule = rule_book.copi_backpropagation
        elif rule == 'ba':
            self.rule = rule_book.copi_broadcast_alignment
        else:
            print(f"rule not supported yet; choose between 'ba' and 'bp'")

    def call(self, inputs, training=None, mask=None, **kwargs):
        xs, acts, ys = [], [], []

        for layer in self.ls:
            x, a, y = layer(inputs)
            inputs = y
            xs.append(x)
            acts.append(a)
            ys.append(y)

        return xs, acts, ys if training else ys

    def train_step(self, data, *args, **kwargs):
        inputs, y_true = data
        batch_size, feature_size = inputs.shape

        # forward pass
        xs, acts, ys = self(inputs, True)

        # backward pass
        rule_info_dict = {
            'df': self.df,
            'acts': acts,
            'ys': ys,
            'y_hat': y_true,
            'alpha_relu': self.alpha,
            'layers': self.ls
        }

        deltas = self.rule(**rule_info_dict)

        # mixed update pass
        for x, a, delta, layer in zip(xs, acts, deltas, self.ls):
            W, R = layer.trainable_variables

            z = a + delta
            q = x @ R
            xT = tf.transpose(x)
            squares = (xT @ x) * tf.eye(x.shape[-1])  # diag(E[x^2])

            dW = (xT @ z - squares @ W) / batch_size
            dR = (xT @ q - squares @ R) / batch_size

            layer.W.assign_add(self.lr_w * dW)
            layer.R.assign_sub(self.lr_r * dR)

        self.compiled_metrics.update_state(y_true, ys[-1])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        ys = self(x)[-1]

        self.compiled_metrics.update_state(y, ys[-1])
        return {m.name: m.result() for m in self.metrics}
