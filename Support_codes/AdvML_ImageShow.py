import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def show_data(dataset, sample):
    data = dataset[sample]
    rgb_image = data[0].numpy().transpose(1, 2, 0)
    mask = data[1].numpy()

    # Create a figure with subplots
    _, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 0.3]})

    # Display the raw RGB image
    axs[0].imshow(rgb_image)
    axs[0].set_title('Raw RGB Image')

    # Create a copy of the RGB image to overlay the masks
    masked_image = rgb_image.copy()

    # Color dictionary to store label colors
    label_list = dataset._class_names
    color_list = dataset._class_colors
    color_dict = {label: tuple(color / 255.0) for label, color in zip(label_list, color_list)}

    # Overlay masks on the second plot and create a legend for labels with at least one pixel
    legend_patches = []
    for idx, label in enumerate(label_list):
        mask_class = (mask == idx)
        if np.sum(mask_class) > 0:
            color = color_dict[label]
            masked_image[mask_class] = color  # Set pixels of the specific class to its color
            legend_patches.append(Patch(color=np.array(color), label=label))

    # Display the masked image
    axs[1].imshow(masked_image)
    axs[1].set_title('Masked Image')

    # Create a legend
    axs[2].legend(handles=legend_patches, title='Labels', loc='center left')
    axs[2].axis('off')  # Turn off axis for the legend

    # Show the plot
    plt.tight_layout()
    plt.show()