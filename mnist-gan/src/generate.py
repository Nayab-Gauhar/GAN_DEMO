"""
Load a saved generator and produce new digit images.

Usage:
    python -m src.generate --model models/generator_model_100.h5 --n-samples 25 --out outputs/sample.png
"""
import argparse
import math
import matplotlib.pyplot as plt
from keras.models import load_model

from src.data import generate_latent_points


def generate_images(model_path, n_samples=25, latent_dim=100, out_path='outputs/sample.png'):
    model = load_model(model_path)
    x_input = generate_latent_points(latent_dim, n_samples)
    X = model.predict(x_input, verbose=0)

    grid = int(math.sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(grid, grid, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved {n_samples} generated images to {out_path}')


def interpolate(model_path, latent_dim=100, steps=10, out_path='outputs/interpolation.png'):
    """Smoothly morph from one random digit into another."""
    model = load_model(model_path)
    v1 = generate_latent_points(latent_dim, 1)[0]
    v2 = generate_latent_points(latent_dim, 1)[0]

    vectors = [v1 + (v2 - v1) * (t / (steps - 1)) for t in range(steps)]
    import numpy as np
    X = model.predict(np.array(vectors), verbose=0)

    for i in range(steps):
        plt.subplot(1, steps, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved interpolation to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate images from a saved GAN generator')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=25)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--out', type=str, default='outputs/sample.png')
    parser.add_argument('--interpolate', action='store_true', help='Generate a morph sequence instead')
    args = parser.parse_args()

    if args.interpolate:
        interpolate(args.model, latent_dim=args.latent_dim, out_path=args.out)
    else:
        generate_images(args.model, n_samples=args.n_samples, latent_dim=args.latent_dim, out_path=args.out)


if __name__ == '__main__':
    main()
