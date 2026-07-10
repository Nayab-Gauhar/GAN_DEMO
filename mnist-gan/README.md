# MNIST DCGAN

A Deep Convolutional GAN (DCGAN) that learns to generate handwritten digit
images by training a generator and discriminator adversarially on the
MNIST dataset.

## How it works

- **Discriminator** — a small CNN classifier that predicts whether an image
  is a real MNIST digit or one produced by the generator.
- **Generator** — takes a 100-dimensional random vector (the "latent
  space") and upsamples it through transposed-convolution layers into a
  28x28 grayscale image.
- **GAN (combined model)** — stacks generator -> discriminator with the
  discriminator's weights frozen, used only to push the generator's
  weights toward fooling the discriminator.

Both networks are trained in alternating steps: the discriminator learns
to tell real from fake, and the generator learns to make fakes the
discriminator can't catch.

## Project structure

```
mnist-gan/
├── src/
│   ├── models.py      # discriminator, generator, gan definitions
│   ├── data.py        # dataset loading + real/fake sample generation
│   ├── train.py        # CLI training entrypoint
│   ├── generate.py    # generate images / interpolations from a saved model
│   └── utils.py        # plotting, checkpointing, loss logging
├── models/              # saved generator checkpoints (.h5)
├── outputs/             # generated image grids + loss_log.csv
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m src.train --epochs 100 --batch-size 256 --checkpoint-every 10
```

This will:
- train for 100 epochs (~234 batches/epoch on the full 60k MNIST training set)
- save a generator checkpoint + sample image grid every 10 epochs into
  `models/` and `outputs/`
- log every batch's discriminator/generator loss to `outputs/loss_log.csv`

**Healthy loss ranges** (from Brownlee's DCGAN tutorial): discriminator
loss around 0.5–0.8, generator loss around 0.5–2. If discriminator loss
collapses toward 0, the generator has likely stopped improving and
training should be restarted.

## Generating new digits from a trained model

```bash
python -m src.generate --model models/generator_model_100.h5 --n-samples 25 --out outputs/sample.png
```

## Latent space interpolation

Morph smoothly from one random digit into another:

```bash
python -m src.generate --model models/generator_model_100.h5 --interpolate --out outputs/interpolation.png
```

## Plotting loss curves

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/loss_log.csv')
plt.plot(df['d_loss'], label='Discriminator')
plt.plot(df['g_loss'], label='Generator')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss_curve.png')
```

## Results

_Add sample generated image grids here after training, e.g. epoch 10 vs
epoch 100 comparison, plus the loss curve plot._

## Reference

Architecture and training procedure based on Jason Brownlee's
[DCGAN MNIST tutorial](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/).
