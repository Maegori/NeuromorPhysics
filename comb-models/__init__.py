from network import LIFNetwork
from grad_functions import surrogate_fn, ba_fn
from utils import poisson_gen

import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )

    # format data and squeeze between 0 and 1
    train_X = train_X.reshape(-1, 784) / 255
    train_y = tf.one_hot(train_y, depth=10, axis=1)
    test_X = test_X.reshape(-1, 784) / 255
    test_y = tf.one_hot(test_y, depth=10, axis=1)

    layer_sizes = [10, 40, 100, 400, 1000]
    step_sizes = [30, 50, 100]

    for time_steps in step_sizes:
        print(f"Number of time steps: {time_steps}")
        for layer_size in layer_sizes:
            print(f"Neurons in hidden layer: {layer_size}")
            for n in range(10):
                # logging
                training_logger = keras.callbacks.CSVLogger(
                    'logs-ba/train_log-time_step{}-layer_size{}-n{}-ba.csv'.format(time_steps, layer_size, n), separator=",", append=True)

                # set up the network
                net = LIFNetwork([layer_size, 10],
                                 nt=time_steps, burn_in=20, grad_fn_name='ba', surrogate_fn=ba_fn, gen=poisson_gen,
                                 dtype=tf.float32)
                net.compile(optimizer='adam', metrics=['mse', 'accuracy'], run_eagerly=True)

                # run the network
                net.fit(train_X, train_y, epochs=10, batch_size=1000, callbacks=[training_logger])
                net.evaluate(test_X, test_y, batch_size=1000)
                net.save_weights('weights-ba/weights-time_step{}-layer_size{}-n{}-ba'.format(time_steps, layer_size, n))