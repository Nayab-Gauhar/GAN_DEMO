
import numpy as np
import streamlit as st
from keras.models import load_model

MODEL_PATH = "models/generator_model_100.h5"
LATENT_DIM = 100


@st.cache_resource
def get_model():
    """Loads the model once and reuses it across reruns/button clicks."""
    return load_model(MODEL_PATH)


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    return x_input.reshape(n_samples, latent_dim)


def generate_digit_grid(model, n_samples=25):
    latent_points = generate_latent_points(LATENT_DIM, n_samples)
    X = model.predict(latent_points, verbose=0)          # shape: (n_samples, 28, 28, 1)
    X = (X * 255).astype(np.uint8)                        # [0,1] float -> [0,255] uint8
    return X


st.title("MNIST Digit Generator")
st.write("A DCGAN trained from scratch on MNIST. Click below to generate new handwritten digits.")

n_samples = st.slider("Number of digits to generate", min_value=1, max_value=25, value=9)

model = get_model()

if st.button("Generate"):
    images = generate_digit_grid(model, n_samples=n_samples)

    cols = st.columns(5)
    for i in range(n_samples):
        with cols[i % 5]:
            # squeeze drops the trailing channel dim: (28,28,1) -> (28,28)
            st.image(images[i].squeeze(), width=100)