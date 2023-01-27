import tensorflow as tf


def ba_fn(v):
    return tf.where(v > 0, (1 / tf.math.cosh(v)) ** 2, 0.)


def surrogate_fn(x):
    s = tf.sigmoid(x)
    return s * (1 - s)


def broadcast_alignment_builder(surrogate_fn):
    def broadcast_alignment_fn(error, layers):
        batch_size = error.shape[0]
        deltas = [error]
        for layer in reversed(layers[:-1]):
            delta = error @ layer.B
            deltas.append(delta)

        grads = []
        for delta, layer in zip(reversed(deltas), layers):
            h = surrogate_fn(layer.state.v)
            iota = h * delta
            grads.append(tf.linalg.einsum('ni,nj->ij', layer.inputs, iota) / batch_size)
            grads.append(tf.reduce_mean(iota, axis=0))
        return grads
    return broadcast_alignment_fn


def surrogate_gradient_builder(surrogate_fn):
    def surrogate_gradient_fn(error, layers):
        batch_size = error.shape[0]
        deltas = [surrogate_fn(layers[-1].state.v) * error]
        for i in reversed(range(0, len(layers) - 1)):
            W = layers[i + 1].W
            d = (deltas[-1] @ tf.transpose(W)) * surrogate_fn(layers[i].state.v)
            deltas.append(d)

        grads = []
        for delta, layer in zip(reversed(deltas), layers):
            x = layer.inputs
            dW = tf.transpose(x) @ delta
            db = tf.reduce_mean(delta, axis=0)
            grads.append(dW)
            grads.append(db)

        return grads

    return surrogate_gradient_fn