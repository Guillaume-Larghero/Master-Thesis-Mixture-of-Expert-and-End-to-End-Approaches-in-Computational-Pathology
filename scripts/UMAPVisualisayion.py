import os
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from models.builder import build_simple_mil_moe
from utils.dataset import H5MultiExpertDatasetInMemory
from utils.helpers import (
    compute_expert_variance,
    create_expert_util,
    create_roc_curve,
    seed_everything,
)
from utils.logger import MetricsLogger
from utils.losses import ZLoss, load_balancing_loss
from models.end2end import CytologyDataset , E2E_MILModel
import random
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cfgs_pth = "/home/gul075/MOE_github/MOE/configs"
cohorts=["BWHQuality","NTU","MRXS"]


def build_E2E_pipeline(cfg: DictConfig) -> E2E_MILModel:
    # Instantiate the VGG feature extractor and ABMIL head using the configuration file
    extractor = instantiate(cfg.extractor)
    head = instantiate(cfg.head)
    
    # Combine them into a single model
    model = E2E_MILModel(extractor, head)
    return model

def get_pooled_representations(model, dataset):
    pooled_reps = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, label in dataset:  
            
            # Extract features from the data using the VGG feature extractor
            vgg_features = model.feature_extractor(data)
            
            # Get the tiles embedding from the ABMIL head
            tiles_emb = model.mil_head.tiles_emb(vgg_features[..., model.mil_head.metadata_cols:], mask = None)
            tiles_emb = tiles_emb.unsqueeze(0)
            pooled_representation, _ = model.mil_head.attention_layer(tiles_emb, mask = None)
            
            pooled_reps.append(pooled_representation.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    return np.vstack(pooled_reps), np.hstack(labels)


@hydra.main(version_base=None, config_path=cfgs_pth, config_name=None)
def main(cfg : DictConfig):
    print("seeding...")
    seed_everything(cfg.seed)
    model = build_E2E_pipeline(cfg)
    print("Model instantiated.")
    weights_path = "/n/scratch/users/g/gul075/checkpoints/Leukemia/AML_APL2_CLIPPED/NTU_SC/40xNORM/clipped/2024-08-09_10-16-47_leukemia_AMLAPL_E2Emil_SC_NTU_150_batch64_NoAug.models.extractor.VGG19FeatureExtractor_models.owkin.abmil.ABMIL_SC_CLIPPEDSAVE_1.pth" # CHANGE THAT
    print(f"weights_path is {weights_path}")
    model.load_state_dict(torch.load(weights_path))
    print("Weights sucesfully loaded!!")
    
    print("building dataset BWH")
    dataset_BWHQuality = CytologyDataset(
                label_file=f"/n/data2/hms/dbmi/kyu/lab/gul075/BWHQuality_Labels_SC_AML_APL.xlsx",
                image_folder=f"/n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_BWHQuality_SC_100x",
                label_column= cfg.data.label_column,
                tile_number=cfg.data.tile_number,
                transform=None,
                augment=False,
            )
    
    print("building dataset NTU")
    dataset_NTU = CytologyDataset(
                label_file=f"/n/data2/hms/dbmi/kyu/lab/gul075/NTU_Labels_SC_AML_APL.xlsx",
                image_folder=f"/n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_NTU_SC_100x",
                label_column= cfg.data.label_column,
                tile_number=cfg.data.tile_number,
                transform=None,
                augment=False,
            )
    
    print("building dataset MRXS")
    dataset_MRXS = CytologyDataset(
                label_file=f"/n/data2/hms/dbmi/kyu/lab/gul075/MRXS_Labels_SC_AML_APL.xlsx",
                image_folder=f"/n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_MRXS_SC_100x",
                label_column= cfg.data.label_column,
                tile_number=cfg.data.tile_number,
                transform=None,
                augment=False,
            )
    
    print("getting pooled rep BWH")
    rep_BWHQuality, labels_BWHQuality = get_pooled_representations(model, dataset_BWHQuality)
    print("getting pooled rep NTU")
    rep_NTU, labels_NTU = get_pooled_representations(model, dataset_NTU)
    print("getting pooled rep MRXS")
    rep_MRXS, labels_MRXS = get_pooled_representations(model, dataset_MRXS)

    # UMAP Plotting
    print("UMAP...")
    pooled_representations = np.vstack([rep_BWHQuality, rep_NTU, rep_MRXS])
    labels = np.hstack([labels_BWHQuality, labels_NTU, labels_MRXS])
    dataset_ids = np.hstack([np.zeros(len(rep_BWHQuality)), 
                             np.ones(len(rep_NTU)), 
                             2*np.ones(len(rep_MRXS))])

    unique_colors = dataset_ids * 2 + labels
    
    print(f"we have {unique_colors} unique colors")
    print(f"and datset ids are : {dataset_ids}")
    
    print("Pooled Representations Statistics:")
    print(f"Shape: {pooled_representations.shape}")
    print(f"Mean: {np.mean(pooled_representations, axis=0)}")
    print(f"Standard Deviation: {np.std(pooled_representations, axis=0)}")
    print(f"Min: {np.min(pooled_representations, axis=0)}")
    print(f"Max: {np.max(pooled_representations, axis=0)}")


    reducer = umap.UMAP(n_neighbors=600 ,min_dist=0.99999999999 , metric = "cosine", spread = 55) # metric = "cosine" or "correlation"
    umap_embeddings = reducer.fit_transform(pooled_representations)
    
    # Ensure the labels are mapped to only 6 unique values
    color_mapping = {
        0: 0, 1: 1,  # Dataset 0 with labels 0 and 1
        2: 2, 3: 3,  # Dataset 1 with labels 0 and 1
        4: 4, 5: 5,  # Dataset 2 with labels 0 and 1
    }
    mapped_colors = np.vectorize(color_mapping.get)(unique_colors)
    
    custom_cmap = ListedColormap(['green', 'red','blue' ,'orange','peru','magenta'])

    # UMAP Plotting
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=mapped_colors, cmap=custom_cmap, s=10)
    plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4, 5])
    plt.title('UMAP of Pooled Representations from ABMIL')

    '''plt.figure(figsize=(10, 7))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=unique_colors, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('UMAP of Pooled Representations from ABMIL')'''

    # Save the plot
    save_path = "/home/gul075/TEMPFIG/UMAP_COSINE_ALLCOHORT_6COLORS_AMLAPL.png"
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
    
    
if __name__ == "__main__":
    main()