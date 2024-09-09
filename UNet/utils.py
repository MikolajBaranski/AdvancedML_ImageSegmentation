import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, random_split

from dataset import SUNRGBDDataset

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    # Make dataset
    dataset_paths = np.load('dataset_paths.npy')
    train_dataset = SUNRGBDDataset(dataset_paths[0], dataset_paths[2], transform=train_transform, split='train')
    test_dataset = SUNRGBDDataset(dataset_paths[0], dataset_paths[2], transform=val_transform, split='test')
    val_dataset = SUNRGBDDataset(dataset_paths[0], dataset_paths[2], transform=val_transform, split='val')

    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # Test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def miou_multilabel(y_true, y_pred, numLabels):
    counter = np.zeros(numLabels)
    iou_list = np.zeros(numLabels)
    for index in range(numLabels):
        intersection = ((y_true == y_pred)&(y_true == index)).sum().item()
        union = (((y_true != 255) & (y_pred==index)) | (y_true == index)).sum().item()
        # Avoid dividing by zero by not including labels that do not appear in the image
        if union != 0:
            iou_list[index] = (intersection / union)
            counter[index] = 1

    if np.sum(counter) != 0:
        miou = np.sum(iou_list) / np.sum(counter)
    else:
        miou = 0
        
    return miou, iou_list, counter

def save_data(image, mask_true, mask_pred, dataset, num):
    rgb_image = image.cpu().numpy().transpose(1, 2, 0)

    # Color dictionary to store label colors
    label_list = dataset._class_names
    color_list = dataset._class_colors
    color_dict = {label: tuple(color / 255.0) for label, color in zip(label_list, color_list)}

    # Create a copy of the RGB image to overlay the masks
    masked_image_true = rgb_image.copy()

    # Overlay masks on the second plot and create a legend for labels with at least one pixel
    legend_patches_true = []
    legend_patches_pred = []
    used_labels = []
    for idx, label in enumerate(label_list):
        mask_class = (mask_true == idx)
        if np.sum(mask_class) > 0:
            color = color_dict[label]
            masked_image_true[mask_class] = color  # Set pixels of the specific class to its color
            used_labels.append(label)
            legend_patches_true.append(Patch(color=np.array(color), label=label))

    # Create a copy of the RGB image to overlay the masks
    masked_image_pred = rgb_image.copy()

    for idx, label in enumerate(label_list):
        mask_class = (mask_pred == idx)
        if np.sum(mask_class) > 0:
            color = color_dict[label]
            masked_image_pred[mask_class] = color  # Set pixels of the specific class to its color
            if label not in used_labels:
              legend_patches_pred.append(Patch(color=np.array(color), label=label))
    
    # Create a figure with subplots
    _, axs = plt.subplots(2, 3, figsize=(11, 7), gridspec_kw={'width_ratios': [1, 1, 0.4], 'hspace': 0.2})

    # Display the raw RGB image
    axs[0][0].imshow(rgb_image)
    axs[0][0].set_title('Raw RGB Image')

    # Display the masked image
    axs[0][1].imshow(masked_image_true)
    axs[0][1].set_title('True Masked Image')

    # Create a legend
    axs[0][2].legend(handles=legend_patches_true, title='True Labels', loc='center left')
    axs[0][2].axis('off')  # Turn off axis for the legend

    # Display the raw RGB image
    axs[1][0].imshow(rgb_image)
    axs[1][0].set_title('Raw RGB Image')

    # Display the masked image
    axs[1][1].imshow(masked_image_pred)
    axs[1][1].set_title('Predicited Masked Image')

    # Create a legend
    axs[1][2].legend(handles=legend_patches_pred, title='False Labels', loc='center left')
    axs[1][2].axis('off')  # Turn off axis for the legend
    
    # Create the save directory if it doesn't exist
    os.makedirs("validation_images", exist_ok=True)
    folder_path = os.path.join("validation_images", f"Epoch_{num[0]}")
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(folder_path, f"plot_e{num[0]}_b{num[1]}_{num[2]}")
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up resources

def check_accuracy(loader, model, dataset, epoch, loss_fn, device="cuda", save_images=False, save_batch=False, num_examples=1):
    print("=> Checking accuracy")
    loss_vals = 0
    num_correct = 0
    num_pixels = 0
    iou_list_score = np.zeros((37,))
    counter_all = np.zeros((37,))
    miou_score = 0
    if (isinstance(save_batch, int) & num_examples==1):
        random_integers = [save_batch]
    else:
        random_integers = np.random.randint(0, len(loader), num_examples)
    
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc="Validation", unit="batch")):
            # Move data to device
            x = x.to(device)
            y = y.to(device)
            # Get predictions and loss values
            preds = model(x)
            loss_vals += loss_fn(preds, y).item()
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            preds = preds.float()
            # Get accuracy and miou score
            num_correct += (preds == y).sum().item()
            num_pixels += torch.sum(y != 255).item()
            miou, iou_list, counter = miou_multilabel(y, preds, 37)
            counter_all += np.array(counter)
            iou_list_score += np.array(iou_list)
            miou_score += miou

            # Save example images
            if (save_images == True) & (idx in random_integers):
              for ydx, (x, pred, y1) in enumerate(zip(x, preds, y)):
                pred_py = pred.cpu().numpy()
                y_py = y1.cpu().numpy()
                save_data(x, y_py, pred_py, dataset,[epoch,idx,ydx])

    # Get average loss
    loss_avg = loss_vals/len(loader)
    # Get accruacy and dice
    iou_score_avg = iou_list_score / counter_all
    miou_score_avg = miou_score/len(loader)
    accuracy_avg = num_correct/num_pixels*100
    
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy_avg:.2f}")
    print(f"MUoI score: {miou_score_avg:.2f}")
    
    model.train()
    
    return loss_avg, accuracy_avg, miou_score_avg, iou_score_avg