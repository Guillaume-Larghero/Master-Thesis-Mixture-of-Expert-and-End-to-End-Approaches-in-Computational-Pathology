import os
import zipfile
import random
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import io

class E2E_MILModel(nn.Module):
    def __init__(self, feature_extractor, mil_head):
        super(E2E_MILModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.mil_head = mil_head

    def forward(self, x):
        batch_size = x.size(0)
        num_tiles = x.size(1)
        x = x.view(batch_size * num_tiles, x.size(2), x.size(3), x.size(4))  # Flatten the batch and tile dimensions (VGG wants batch_size*num_tiles, channel, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_tiles, -1)  # Reshape back to (batch_size, num_tiles, feature_dim)
        logits = self.mil_head(features)
        return logits
    
    
class CytologyDataset(Dataset):
    def __init__(self, label_file, image_folder, label_column, tile_number, transform=None, augment=False , index_list = None):
        """
        Args:
            label_file (string): Path to the Excel file with annotations.
            image_folder (string): Directory with all the .zip files containing images.
            label_column (string): The column name in the Excel file that contains the labels.
            tile_number (int): Number of tiles to select per sample.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool): If true, augment the dataset by adding transformed versions of the tiles.
        """
        self.labels_df = pd.read_excel(label_file)
        if index_list is not None:
            self.labels_df = self.labels_df.iloc[index_list].reset_index(drop=True)
            
        self.image_folder = image_folder
        self.label_column = label_column
        self.tile_number = tile_number
        self.transform = transform if transform is not None else CytologyTransform()
        self.augment = augment
        if self.augment:
            self.augmentation_transform = AugmentationTransform()
            
        self.slide_ids = self.labels_df['Slide_ID'].tolist()
        self.compute_weights()
        
    def get_label(self, slide_id):
        """
        Get the label for a given slide_id.
        """
        return self.labels_df.loc[self.labels_df['Slide_ID'] == slide_id, self.label_column].values[0]
    
    def compute_weights(self):
        """
        Compute weights for WeightedRandomSampler.
        """
        class_counts = {}

        # Count the number of occurrences of each class
        for slide_id in self.slide_ids:
            label = self.get_label(slide_id)
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # Compute the weight for each class
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

        # Assign weights to each sample based on its label
        self.weights = []
        for slide_id in self.slide_ids:
            label = self.get_label(slide_id)
            self.weights.append(class_weights[label])
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        slide_id = self.labels_df.iloc[idx]['Slide_ID']
        label = self.labels_df.iloc[idx][self.label_column]
        zip_path = os.path.join(self.image_folder, f"{slide_id}.zip")
        
        images = []
        with zipfile.ZipFile(zip_path, 'r') as archive:
            image_files = [file for file in archive.namelist() if file.endswith('.png')]
            selected_files = random.sample(image_files, min(self.tile_number, len(image_files)))
            
            for image_file in selected_files:
                with archive.open(image_file) as file:
                    img = Image.open(io.BytesIO(file.read())).convert('RGB')  # Ensure RGB format
                    img_transfo = self.transform(img)
                    images.append(img_transfo)
                    
                    if self.augment:
                        augmented_img = self.augmentation_transform(img)
                        images.append(augmented_img)
        
        # If not enough images, pad with zeros
        while len(images) < self.tile_number * (2 if self.augment else 1):
            images.append(torch.zeros_like(images[0]))
        
        images = torch.stack(images)  # Shape: (tile_number, channels, height, width)
        label = torch.tensor(label, dtype=torch.float32)
        
        return images, label
    
class CytologyTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img):
        return self.transform(img)
    
class AugmentationTransform:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1)
            ]),
            transforms.ColorJitter(
                brightness=0.15,  # Adjust brightness ±15%
                contrast=0.15,
                saturation=0.15,
                hue=0.1
            ),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img):
        return self.augment(img)
    
    
    
    
'''# Example usage
dataset = CytologyDataset(
    label_file='/n/data2/hms/dbmi/kyu/lab/gul075/NTU_Labels.xlsx',
    image_folder='/n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_NTU_SC_100x',
    label_column='AML_APL',  # Change this to 'FLT3' or 'NPM1' as needed
    tile_number=1000
)'''