from collections import namedtuple

import tensorflow as tf
import numpy as np


DATA_SAMPLES = {
    'mnist': [200, 300, 500, 700, 923, 1235, 9080, 998, 77, 71],
    'fashion_mnist': [2023, 30, 1500, 47, 93, 135, 80, 99, 77, 71],
    'mnist-fashion-mnist-mixed': [2023, 30, 1500, 47, 93, 10135, 12023, 10030, 11500, 10047],
}

DataIterators = namedtuple('DataIterators',
                           ['x', 'batch_size', 'z', 'x_iter', 'z_iter', 'train_init_op', 'test_init_op', 'z_init_op'])


def prepare_data(name):
    if name == 'mnist-fashion-mnist-mixed':
        mnist = tf.keras.datasets.mnist
        fmnist = tf.keras.datasets.fashion_mnist

        (mx_train, my_train), (mx_test, my_test) = mnist.load_data()
        (fx_train, fy_train), (fx_test, fy_test) = fmnist.load_data()

        x_train = np.vstack((mx_train, fx_train))
        x_test = np.vstack((mx_test, fx_test))
        y_train = np.hstack((my_train, fy_train+10))
        y_test = np.hstack((my_test, fy_test+10))

    else:
        data = getattr(tf.keras.datasets, name)
        (x_train, y_train), (x_test, y_test) = data.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # shuffle training data
    np.random.seed(71)
    rand_indices = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[rand_indices, :], y_train[rand_indices]

    return x_train, y_train, x_test, y_test


def build_tf_data_iterators(x_dims, latent_dims):
    ph_x = tf.placeholder(tf.float32, [None, x_dims], name='input')
    ph_batch_size = tf.placeholder(tf.int64, name='batch_size')
    ph_z = tf.placeholder(tf.float32, [None, latent_dims], name='z_input')

    train_dataset = tf.data.Dataset.from_tensor_slices(ph_x).batch(ph_batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(ph_x).batch(ph_batch_size)

    x_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_iterator_init_op = x_iterator.make_initializer(train_dataset)
    test_iterator_init_op = x_iterator.make_initializer(test_dataset)

    z_dataset = tf.data.Dataset.from_tensor_slices(ph_z).batch(ph_batch_size)
    z_iterator = tf.data.Iterator.from_structure(z_dataset.output_types, z_dataset.output_shapes)

    z_iterator_init_op = z_iterator.make_initializer(z_dataset)

    return DataIterators(x=ph_x, z=ph_z, batch_size=ph_batch_size, x_iter=x_iterator, z_iter=z_iterator,
                         train_init_op=train_iterator_init_op,
                         test_init_op=test_iterator_init_op, z_init_op=z_iterator_init_op)
