from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random


def showImages(dataset: Dataset, title: str):
    # Choose 25 random indices from the dataset
    selected_indices = random.sample(range(len(dataset)), 25)
    images = []

    # Iterate over the selected indices and retrieve the corresponding images
    for index in selected_indices:
        image, _ = dataset[index]
        images.append(image[0, :, :])

    # Create a 5x5 subplot grid
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        # Get a random index
        img = images[i]  # Shape: (128, 128)
        # Display the image
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)  # Add title to the figure
    plt.show()
