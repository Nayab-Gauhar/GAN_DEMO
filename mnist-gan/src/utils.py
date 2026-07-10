"""
Helper functions: saving generated-image plots, evaluating discriminator
performance, and saving checkpoints. Also a small CSV logger for loss curves.
"""
import csv
import os
import matplotlib.pyplot as plt

from src.data import generate_real_samples, generate_fake_samples


def save_plot(examples, epoch, n=10, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = os.path.join(out_dir, 'generated_plot_e%03d.png' % (epoch + 1))
    plt.savefig(filename)
    plt.close()
    return filename


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim,
                           n_samples=100, model_dir='models', out_dir='outputs'):
    os.makedirs(model_dir, exist_ok=True)

    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'Epoch: {epoch+1} | Accuracy Real: {acc_real*100:.2f}% | Accuracy Fake: {acc_fake*100:.2f}%')

    save_plot(x_fake, epoch, out_dir=out_dir)

    filename = os.path.join(model_dir, 'generator_model_%03d.h5' % (epoch + 1))
    g_model.save(filename)

    return acc_real, acc_fake


class LossLogger:
    """Appends per-batch d_loss/g_loss to a CSV so you can plot curves later."""

    def __init__(self, path='outputs/loss_log.csv'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'd_loss', 'g_loss'])

    def log(self, epoch, batch, d_loss, g_loss):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, d_loss, g_loss])
