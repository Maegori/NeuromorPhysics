import tensorflow as tf


def copi_backpropagation(**kwargs):
    """ returns all deltas for dloss/da_l """
    df = kwargs['df']
    acts = kwargs['acts']
    ys = kwargs['ys']
    y_hat = kwargs['y_hat']
    alpha = kwargs['alpha_relu']
    layers = kwargs['layers']

    d = -df(acts[-1], alpha) * (ys[-1] - y_hat)
    deltas = [d]
    for i in reversed(range(0, len(layers) - 1)):
        W = layers[i + 1].W
        R = layers[i + 1].R

        d = (d @ tf.transpose(W) @ tf.transpose(R)) * df(acts[i], alpha)
        deltas.append(d)
    return reversed(deltas)


def copi_broadcast_alignment(**kwargs):
    """ returns all deltas for dloss/da_l using a random feedback matrix B """
    ys = kwargs['ys']
    y_hat = kwargs['y_hat']
    layers = kwargs['layers']

    error = y_hat - ys[-1]

    deltas = []
    for layer in layers[:-1]:
        feedback = error @ layer.B
        deltas.append(feedback)
    deltas.append(error)

    return deltas