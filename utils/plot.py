import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

mnist_cmap = cm.get_cmap('brg', 10)


def scatter_plot(data, labels, title="Latent Space", filename=""):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=mnist_cmap)
    plt.ylabel('z2')
    plt.xlabel('z1')
    plt.title(title)
    plt.colorbar()

    if filename:
        plt.savefig('%s.png' % filename)

    plt.close(fig)


def latent_interpolate(sess, model, data_iterators, title="Latent-Space Interpolation", total_samples=20, boundary=None,
                       filename=None):

    a = np.linspace(*boundary[0], total_samples)
    b = np.linspace(*boundary[1], total_samples)
    xx, yy = np.meshgrid(a, b)

    latent_space = np.vstack([xx.T.reshape(-1), yy.T.reshape(-1)]).T

    sess.run(data_iterators.z_init_op,
             feed_dict={data_iterators.z: latent_space, data_iterators.batch_size: total_samples ** 2})

    _, likelihoods = model.decoder(data_iterators.z_iter.get_next())
    x = sess.run(likelihoods).reshape((-1, 28, 28))

    fig = plt.figure(figsize=(10, 10))
    img = np.zeros((total_samples * 28, total_samples * 28))

    for i in range(0, total_samples * 28, 28):
        for j in range(0, total_samples * 28, 28):
            img[(total_samples * 28 - j - 28):(total_samples * 28 - j), i:i + 28] = x[
                                                                                    i // 28 * total_samples + j // 28,
                                                                                    :, :]

    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title(title)
    if filename:
        plt.savefig('%s.png' % filename)
    plt.close(fig)


def reconstruction_progress_plot(data, no_epoches, filename=None):
    if len(data) > 10:
        data[9] = data[-1]
        data = data[:10]
    fig = plt.figure(figsize=(10, 10))

    for i in range(10):
        for j in range(len(data)):
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.imshow(data[j][i].reshape((28, 28)), cmap='gray')
            plt.axis('off')
            if i == 0 and j == 0:
                plt.title('Data')
            elif i == 0 and j == 5:
                plt.title('Reconstruction at different epoches (%d)' % no_epoches)
    if filename:
        plt.savefig(filename)

    plt.close(fig)

