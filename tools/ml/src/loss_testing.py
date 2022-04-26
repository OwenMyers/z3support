import tf_vae
import numpy as np


def main():
    x = np.zeros([3, 2, 2, 1]) + 1.0
    y = np.zeros([3, 2, 2, 1]) + 0.5
    loss = tf_vae.gl_vae_r_loss(x, y)
    print(f"loss: {loss}")

if __name__ == "__main__":
    main()