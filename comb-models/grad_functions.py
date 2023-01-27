import tensorflow as tf

# surrogate gradient function for broadcast alignment (can also be used for surrogate gradient algorithm)
def ba_fn(v):
    return tf.where(v > 0, (1 / tf.math.cosh(v)) ** 2, 0.)


# sigmoid derivative for surrogate gradient algorithm
def surrogate_fn(x):
    s = tf.sigmoid(x)
    return s * (1 - s)

def broadcast_alignment_builder(surrogate_fn):
    """ return the broadcast alignment function using the surrogate gradient """
    def broadcast_alignment_fn(error, layers):
        batch_size = error.shape[0]
        deltas = [error]
        # backprop the error
        for layer in reversed(layers[:-1]):
            delta = error @ layer.B
            deltas.append(delta)

        # compute gradient for each parameter in the network
        grads = []
        for delta, layer in zip(reversed(deltas), layers):
            h = surrogate_fn(layer.state.v)
            iota = h * delta
            grads.append(tf.linalg.einsum('ni,nj->ij', layer.inputs, iota) / batch_size)
            grads.append(tf.reduce_mean(iota, axis=0))
        return grads
    return broadcast_alignment_fn


def surrogate_gradient_builder(surrogate_fn):
    """ return the surrogate gradient function using the surrogate gradient """

    def surrogate_gradient_fn(error, layers):
        # backpropagate the error
        deltas = [surrogate_fn(layers[-1].state.v) * error]
        for i in reversed(range(0, len(layers) - 1)):
            W = layers[i + 1].W
            d = (deltas[-1] @ tf.transpose(W)) * surrogate_fn(layers[i].state.v)
            deltas.append(d)

        # compute the gradient for all parameters in the network
        grads = []
        for delta, layer in zip(reversed(deltas), layers):
            x = layer.inputs
            dW = tf.transpose(x) @ delta
            db = tf.reduce_mean(delta, axis=0)
            grads.append(dW)
            grads.append(db)

        return grads

    return surrogate_gradient_fn