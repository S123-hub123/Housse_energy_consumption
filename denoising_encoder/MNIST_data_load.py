import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file from your specific path
data = np.load('/Users/soukaina/Desktop/mnist.npz')

# Access the training images and labels
x_train = data['x_train']
y_train = data['y_train']

# Loop through all images
for i in range(len(x_train)):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')  # Hide the axes
    plt.show()  # Display the image

    # Pause and wait for user input to proceed to the next image
    input("Press Enter to see the next image...")


