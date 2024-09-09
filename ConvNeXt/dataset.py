from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms

from sunrgbd_dataset import SUNRBDBase

class SUNRGBDDataset(Dataset, SUNRBDBase):
    def __init__(
        self, 
        rgb_paths, 
        label_paths,
        transform=None,
        split=None,
        test_size=0.2,
        val_size=0.2,
        random_seed=42):

        # Split the data into train, test, and validation sets
        
        if split == None:
            self.rgb_paths = rgb_paths
            self.label_paths = label_paths

        else:
            train_rgb_paths, test_rgb_paths, train_label_paths, test_label_paths = train_test_split(
                rgb_paths, label_paths, test_size=test_size, random_state=random_seed
            )

            train_rgb_paths, val_rgb_paths, train_label_paths, val_label_paths = train_test_split(
                train_rgb_paths, train_label_paths, test_size=val_size, random_state=random_seed
            )

            if split == 'train':
                self.rgb_paths = train_rgb_paths
                self.label_paths = train_label_paths
            elif split == 'test':
                self.rgb_paths = test_rgb_paths
                self.label_paths = test_label_paths
            elif split == 'val':
                self.rgb_paths = val_rgb_paths
                self.label_paths = val_label_paths
            else:
                raise ValueError("Invalid split argument. Use 'train', 'test', or 'val'.")
            
        self.transform = transform

        self._n_classes = self.N_CLASSES
        self._class_names = self.CLASS_NAMES_ENGLISH[1:]
        self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')[1:]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):

        # Load RGB and depth images
        rgb_image = Image.open(self.rgb_paths[idx]).convert("RGB")
        # Convert the PIL image to a NumPy array
        image = np.array(rgb_image)
    
        # Load label mask
        mask = np.load(self.label_paths[idx])

        # Apply transformations
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        # Reduce each value by 1
        mask -= 1

        # Replace all values that are -1 with 255
        mask[mask == -1] = 255

        # Convert NumPy arrays to PyTorch tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()

        return image , mask