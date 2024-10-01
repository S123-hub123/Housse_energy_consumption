import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Simulate 100 grayscale images (you can replace this with real images)
def generate_synthetic_images(n_images, size=(64, 64)):
    return np.random.rand(n_images, size[0], size[1])

train_images = generate_synthetic_images(100)
test_images = generate_synthetic_images(20)

# Normalize images to [0, 1] range
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

# Add an extra dimension for channels (grayscale -> 1 channel)
train_images = np.reshape(train_images, (len(train_images), 64, 64, 1))
test_images = np.reshape(test_images, (len(test_images), 64, 64, 1))



# Add uniform noise to the images for training
def add_uniform_noise(images, low=-0.5, high=0.5):
    noise = np.random.uniform(low, high, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0., 1.)  # Clip to [0, 1] range
    return noisy_images

train_noisy = add_uniform_noise(train_images)



def build_denoising_autoencoder():
    # Encoder
    input_img = layers.Input(shape=(64, 64, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    encoded = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Compile the model
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

autoencoder = build_denoising_autoencoder()




autoencoder.fit(train_noisy, train_images,  # Noisy images as input, clean as target
                epochs=50,
                batch_size=16,
                shuffle=True,
                validation_split=0.1)



def add_gaussian_noise(images, mean=0., std=0.5):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0., 1.)  # Clip to [0, 1]
    return noisy_images

test_noisy = add_gaussian_noise(test_images)




denoised_images = autoencoder.predict(test_noisy)

# Visualize original, noisy, and denoised images
def plot_images(original, noisy, denoised, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        # Original images
        plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(64, 64), cmap='gray')
        plt.axis('off')

        # Noisy images
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].reshape(64, 64), cmap='gray')
        plt.axis('off')

        # Denoised images
        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(denoised[i].reshape(64, 64), cmap='gray')
        plt.axis('off')

    plt.show()

plot_images(test_images, test_noisy, denoised_images)
