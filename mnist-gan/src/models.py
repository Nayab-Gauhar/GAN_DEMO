"""
Model definitions for MNIST DCGAN.
- define_discriminator(): real vs fake classifier
- define_generator(): latent vector -> 28x28 image
- define_gan(): stacks generator + discriminator (for training generator only)
"""
from keras.layers import Dense, Dropout, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential


def define_discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def define_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model


def define_gan(g_model, d_model):
    # freeze discriminator weights when training via the combined model
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
    return model
