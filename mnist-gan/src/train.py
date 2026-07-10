"""
Train the MNIST DCGAN.

Usage:
    python -m src.train --epochs 100 --batch-size 256 --latent-dim 100
"""
import argparse
import numpy as np

from src.models import define_discriminator, define_generator, define_gan
from src.data import load_real_samples, generate_real_samples, generate_fake_samples, generate_latent_points
from src.utils import summarize_performance, LossLogger


def train(gan_model, d_model, g_model, dataset, latent_dim,
          n_epochs=100, n_batch=256, checkpoint_every=10,
          model_dir='models', out_dir='outputs'):
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    logger = LossLogger(path=f'{out_dir}/loss_log.csv')

    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            d_loss, _ = d_model.train_on_batch(X, y)

            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            logger.log(i + 1, j + 1, d_loss, g_loss)
            print(f'Epoch: {i+1} | Batch: {j+1}/{batch_per_epoch} | D_Loss: {d_loss:.3f} | G_Loss: {g_loss:.3f}')

        if (i + 1) % checkpoint_every == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim,
                                   model_dir=model_dir, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(description='Train MNIST DCGAN')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--out-dir', type=str, default='outputs')
    args = parser.parse_args()

    d_model = define_discriminator()
    g_model = define_generator(args.latent_dim)
    gan_model = define_gan(g_model, d_model)
    dataset = load_real_samples()

    train(gan_model, d_model, g_model, dataset, args.latent_dim,
          n_epochs=args.epochs, n_batch=args.batch_size,
          checkpoint_every=args.checkpoint_every,
          model_dir=args.model_dir, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
