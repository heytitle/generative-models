import tensorflow as tf
import numpy as np
import os
import autoencoders
import shutil
import imageio
import glob

from tqdm import tqdm
from utils import data_provider, plot

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('latent_dims', 2, 'Number of latent dimensions.')
flags.DEFINE_integer('epoch', 20, 'Number of epoch.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_string('model', 'VAE', 'Model to train.')
flags.DEFINE_string('dataset', 'mnist', 'Dataset to train.')
flags.DEFINE_boolean('animation', False, 'If yes, animation of latent space interpolation will be created')

def main(_):
    # prepare output dir
    output_path = 'output/%s' % ('-'.join([FLAGS.model, FLAGS.dataset]))
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    path = lambda x: '%s/%s' % (output_path, x)

    if FLAGS.animation and not os.path.exists(path('gif')):
        os.mkdir(path('gif'))

    # prepare dataset
    x_train, _, x_test, y_test = data_provider.prepare_data(FLAGS.dataset)

    data_iterators = data_provider.build_tf_data_iterators(x_train.shape[1], FLAGS.latent_dims)

    x_iter = data_iterators.x_iter.get_next()

    # prepare model and training
    model = getattr(autoencoders, FLAGS.model)(x_iter, FLAGS.latent_dims)

    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss)

    data_samples = [x_test[data_provider.DATA_SAMPLES[FLAGS.dataset]]]
    data_sampling_interval = np.max([FLAGS.epoch // 8, 1])

    total_batches = int(np.ceil(x_train.shape[0] / FLAGS.batch_size))

    loss, val_loss = (0, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with tqdm(total=FLAGS.epoch*total_batches) as pbar:
            pbar.set_description('Training %s' % FLAGS.model)

            for i in range(FLAGS.epoch):
                sess.run(data_iterators.train_init_op,
                         feed_dict={data_iterators.x: x_train, data_iterators.batch_size: FLAGS.batch_size})

                try:
                    while True:
                        _, loss = sess.run([train_op, model.loss])
                        if pbar.n % 100 == 0:
                            pbar.set_postfix(loss='%.2f' % loss, val_loss='%.2f' % val_loss)
                        pbar.update()

                except tf.errors.OutOfRangeError:
                    pass

                sess.run(data_iterators.test_init_op,
                         feed_dict={data_iterators.x: x_test, data_iterators.batch_size: x_test.shape[0]})
                val_loss = sess.run(model.loss)

                pbar.set_postfix(loss='%.2f' % loss, val_loss='%.2f' % val_loss)

                if FLAGS.animation and (i < 10 or i % 2000 == 0):
                    plot.latent_interpolate(sess, model, data_iterators, boundary=[
                            (-15, 15),
                            (-15, 15),
                        ],
                        title='%s Latent Space Interpolation' % FLAGS.model,
                        filename=path('gif/latent-space-step-%s' % str(i).rjust(5, '0'))
                    )

                if i % data_sampling_interval == 0:
                    sess.run(data_iterators.test_init_op,
                             feed_dict={data_iterators.x: data_samples[0],
                                        data_iterators.batch_size: len(data_samples[0])})

                    data_samples.append(sess.run(model.likelihoods))

        print('Producing artifacts')

        plot.reconstruction_progress_plot(data_samples, filename=path('reconstruction-progress'),
                                          no_epoches=FLAGS.epoch)

        sess.run(data_iterators.test_init_op,
                 feed_dict={data_iterators.x: x_test, data_iterators.batch_size: x_test.shape[0]})

        test_z_samples = sess.run(model.z)

        z_boundary = [
            (np.min(test_z_samples[:, 0]), np.max(test_z_samples[:, 0])),
            (np.min(test_z_samples[:, 1]), np.max(test_z_samples[:, 1]))
        ]

        plot.scatter_plot(test_z_samples, y_test, title='%s Latent Space' % FLAGS.model,
                          filename=path('latent-space-scatter-plot'))


        plot.latent_interpolate(sess, model, data_iterators, boundary=z_boundary,
                                title='%s Latent Space Interpolation' % FLAGS.model,
                                filename=path('latent-space-interpolation'))

        if FLAGS.animation:
            imageio.mimsave(path('animation-latent-space-interpolation.gif'),
                            map(lambda x: imageio.imread(x), sorted(glob.glob(path('gif/*.png')))),
                            duration=0.1)

            shutil.rmtree(path('gif'))

if __name__ == '__main__':
    tf.app.run(main)
