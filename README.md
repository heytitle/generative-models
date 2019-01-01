# Generative Models

This repository contains the reimplementation of generative models. Results from these models can be found in `./output`.

### Usage (TODO)
```
train.py:
  --[no]animation: If yes, animation of latent space interpolation will be created
    (default: 'false')
  --batch_size: Batch size.
    (default: '64')
    (an integer)
  --dataset: [mnist, fashion_mnist, mnist-fashion-mnist-mixed]
    (default: 'mnist')
  --epoch: Number of epoch.
    (default: '20')
    (an integer)
  --latent_dims: Number of latent dimensions.
    (default: '2')
    (an integer)
  --learning_rate: Initial learning rate.
    (default: '0.01')
    (a number)
  --model: Model to train.
    (default: 'VAE')
```

### Autoencoders
- [Variational Autoencoder (VAE)][vae]

### Implementaion Ideas
- [[1406.5298] Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)
  - [GitHub - wohlert/semi-supervised-pytorch: Implementations of different VAE-based semi-supervised and generative models in PyTorch](https://github.com/wohlert/semi-supervised-pytorch)
- [[1506.02216] A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216)
- [[1502.04623] DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623)
- [[1410.6460] Markov Chain Monte Carlo and Variational Inference: Bridging the Gap](https://arxiv.org/abs/1410.6460)

## Acknowledgements
- This repository is inspired by @wiseodd's [generative-models repository](https://github.com/wiseodd/generative-models).
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [Variational Autoencoder and Extensions - VideoLectures.NET](http://videolectures.net/deeplearning2015_courville_autoencoder_extension/?q=variational%20autoencoder)

[vae]: https://arxiv.org/abs/1312.6114

