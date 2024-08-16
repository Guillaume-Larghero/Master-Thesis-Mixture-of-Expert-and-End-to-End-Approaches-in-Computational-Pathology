import os
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil


from models.extractor import VGG19FeatureExtractor
from models.end2end import CytologyDataset , E2E_MILModel
from utils.helpers import (
    create_roc_curve,
    seed_everything,
)
from utils.logger import MetricsLogger

cfgs_pth = "/home/gul075/MOE_github/MOE/configs"
cohorts=["BWHQuality" ,"NTU","MRXS"]
metricfile = "/n/data2/hms/dbmi/kyu/lab/gul075/SC_Metrics_clipped.csv"

def build_E2E_pipeline(cfg: DictConfig) -> E2E_MILModel:
    # Instantiate the VGG feature extractor and ABMIL head using the configuration file
    extractor = instantiate(cfg.extractor)
    head = instantiate(cfg.head)
    
    # Combine them into a single model
    model = E2E_MILModel(extractor, head)
    return model

def get_datasets_for_fold(cfg, fold):
    # Load the label file
    labels_df = pd.read_excel(cfg.data.label_file)
    
    # Stratified K-Fold for splitting the data into train and test
    skf = StratifiedKFold(n_splits=cfg.data.folds, shuffle=True, random_state=cfg.seed)
    
    # Get the indices for the train and test split for the current fold
    splits = list(skf.split(labels_df['Slide_ID'], labels_df[cfg.data.label_column]))
    train_idx, test_idx = splits[fold]
    
    # Get the training labels and indices
    train_labels = labels_df.iloc[train_idx][cfg.data.label_column]
    train_slide_ids = labels_df.iloc[train_idx]['Slide_ID']
    
    # Stratified split for validation set within the training set
    train_idx_strat, val_idx_strat = train_test_split(
        train_idx, test_size=0.25, stratify=train_labels, random_state=cfg.seed
    )
    
    # Create the datasets using the index_list argument
    train_dataset = CytologyDataset(
        label_file=cfg.data.label_file,
        image_folder=cfg.data.image_folder,
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=train_idx_strat
    )
    
    val_dataset = CytologyDataset(
        label_file=cfg.data.label_file,
        image_folder=cfg.data.image_folder,
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=val_idx_strat  
    )
    
    test_dataset = CytologyDataset(
        label_file=cfg.data.label_file,
        image_folder=cfg.data.image_folder,
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=test_idx
    )
    
    print(f"Length of train dataset: {len(train_dataset)} with weights {train_dataset.weights}")
    print(f"Length of val dataset: {len(val_dataset)} with weights {val_dataset.weights}")
    print(f"Length of test dataset: {len(test_dataset)} with weights {test_dataset.weights}")
    
    return train_dataset, val_dataset, test_dataset

def train_one_epoch(
    cfg: DictConfig,
    model: nn.Module,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    logger: MetricsLogger,
):
    accumulation_steps =cfg.data.complete_batch_size // cfg.data.batch_size  # Effective batch size =32, small batch size = 4
    
    use_mixed_precision = False
    scaler = GradScaler() if use_mixed_precision else None

    all_labels = []
    all_preds = []
    all_probs = []
    bar = tqdm(dl, total=len(dl), desc=f"Train Epoch {epoch}")
    model.train()

    optimizer.zero_grad()  # Initialize gradients

    for i, batch in enumerate(bar):
        if len(batch[0]) == 0:
            continue
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
    
        if use_mixed_precision:
            with autocast():  # Mixed precision context
                # Forward pass
                output_logits = model(images)
                output_probs = torch.sigmoid(output_logits)
                preds = (output_probs >= 0.5).long()

                # Calculate loss
                loss = criterion(output_logits, labels.unsqueeze(1))
                loss = loss / accumulation_steps  # Scale loss for gradient accumulation

            scaler.scale(loss).backward()  # Backward pass with scaled loss

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dl):  # Update on the last batch
                scaler.step(optimizer)  # Update model parameters
                scaler.update()  # Update scaler
                optimizer.zero_grad()  # Reset gradients for next accumulation cycle
        else:
            # Regular precision context
            # Forward pass
            output_logits = model(images)
            output_probs = torch.sigmoid(output_logits)
            preds = (output_probs >= 0.5).long()

            # Calculate loss
            loss = criterion(output_logits, labels.unsqueeze(1))
            loss = loss / accumulation_steps  # Scale loss for gradient accumulation

            loss.backward()  # Backward pass

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dl):  # Update on the last batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # Reset gradients for next accumulation cycle

        # Metrics calculation
        acc = (preds.squeeze() == labels).float().mean()
        all_labels.extend(labels.cpu().numpy())
        
        # Handle different cases for preds
        preds_np = preds.squeeze().cpu().numpy()

        if preds_np.ndim == 0:  # If preds_np is a scalar (0-d array)
            all_preds.append(preds_np.item())
        else:
            all_preds.extend(preds_np)


        '''if preds_np.ndim == 0:  # If preds_np is a scalar (0-d array)
            all_preds.append(preds_np.item())
        else:
            all_preds.extend(preds_np)
        
        if isinstance(preds, torch.Tensor):
            if preds.dim() > 0:  # Check if preds is a multi-dimensional tensor
                all_preds.extend(preds.squeeze().cpu().numpy())
            else:  # Scalar tensor
                all_preds.append(preds.item())
        elif isinstance(preds, np.ndarray):
            if preds.ndim > 0:  # Check if preds is a multi-dimensional array
                all_preds.extend(preds.squeeze())
            else:  # Scalar array
                all_preds.append(preds.item())
        else:
            raise TypeError(f"Unexpected type for preds: {type(preds)}")'''
            
        all_probs.extend(output_probs.detach().cpu().numpy())
        
        logger.log_dict({"train/loss": loss.item() * accumulation_steps})  # Log the unscaled loss
        logger.log_dict({"train/acc": acc.item()})

    # Final check for remaining gradients if not updated
    if (i + 1) % accumulation_steps != 0:
        if use_mixed_precision:
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update scaler
        else:
            optimizer.step()  # Update model parameters
        optimizer.zero_grad()  # Reset gradients for next accumulation cycle

    # Epoch metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"all labels are {all_labels}")
    print(f"all probs are {all_probs[:, 0]}")
    
    
    thresholds = np.arange(0.0, 1.0, 0.0001)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, average="weighted")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # compute metrics with best threshold:
    print(f"Best thresh for training epoch is {best_threshold}")
    final_preds = (all_probs >= best_threshold).astype(int)
    print(f"and all preds at best tresh are {final_preds}")
    
    balanced_acc = balanced_accuracy_score(all_labels, final_preds)
    mcc = matthews_corrcoef(all_labels, final_preds)
    weighted_f1 = f1_score(all_labels, final_preds, average="weighted")
    macro_f1 = f1_score(all_labels, final_preds, average="macro")
    
    if cfg.data.n_classes == 2 and len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 0])
        auc_pr = average_precision_score(all_labels, all_probs[:, 0])
        curve_np = create_roc_curve(all_labels, all_probs[:, 0])
        logger.log_image("train/roc_curve", curve_np)
    else:
        auc_pr = 0.1
        roc_auc = 0.5
    
    logger.log_dict({"train/balanced_acc": balanced_acc})
    logger.log_dict({"train/roc_auc": roc_auc})
    logger.log_dict({"train/mcc": mcc})
    logger.log_dict({"train/weighted_f1": weighted_f1})
    logger.log_dict({"train/macro_f1": macro_f1})
    logger.log_dict({"train/auc_pr": auc_pr})
    
    metrics = {
        "roc_auc": roc_auc,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "auc_pr": auc_pr,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }
    return metrics

def val_one_epoch(
    cfg: DictConfig,
    model: nn.Module,
    dl: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: MetricsLogger,
):
    use_mixed_precision = False

    all_labels = []
    all_preds = []
    all_probs = []
    bar = tqdm(dl, total=len(dl), desc=f"Val Epoch {epoch}")
    model.eval()

    with torch.no_grad():
        for batch in bar:
            if len(batch[0]) == 0:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            if use_mixed_precision:
                with autocast():  # Mixed precision context
                    # Forward pass
                    output_logits = model(images)
                    output_probs = torch.sigmoid(output_logits)
                    preds = (output_probs >= 0.5).long()

                    # Calculate loss
                    loss = criterion(output_logits, labels.unsqueeze(1))
            else:
                # Regular precision context
                # Forward pass
                output_logits = model(images)
                output_probs = torch.sigmoid(output_logits)
                preds = (output_probs >= 0.5).long()

                # Calculate loss
                loss = criterion(output_logits, labels.unsqueeze(1))

            # Metrics calculation
            acc = (preds.squeeze() == labels).float().mean()
            all_labels.extend(labels.cpu().numpy())
            if len(preds.shape) > 0:
                all_preds.extend(preds.squeeze().cpu().numpy())
            else:
                all_preds.extend(preds.cpu().numpy())
            all_probs.extend(output_probs.detach().cpu().numpy())
            
            logger.log_dict({"val/loss": loss.item()})
            logger.log_dict({"val/acc": acc.item()})

        # Epoch metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        print(f"all labels are {all_labels}")
        print(f"all probs are {all_probs[:, 0]}")
        
        thresholds = np.arange(0.0, 1.0, 0.0001)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            preds = (all_probs >= threshold).astype(int)
            f1 = f1_score(all_labels, preds, average="weighted")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # compute metrics with best threshold:
        print(f"Best thresh for validation epoch is {best_threshold}")
        final_preds = (all_probs >= best_threshold).astype(int)
        print(f"and all preds at best tresh are {final_preds}")
        
        balanced_acc = balanced_accuracy_score(all_labels, final_preds)
        mcc = matthews_corrcoef(all_labels, final_preds)
        weighted_f1 = f1_score(all_labels, final_preds, average="weighted")
        macro_f1 = f1_score(all_labels, final_preds, average="macro")

        if cfg.data.n_classes == 2 and len(np.unique(all_labels)) > 1:
            auc_pr = average_precision_score(all_labels, all_probs[:, 0])
            roc_auc = roc_auc_score(all_labels, all_probs[:, 0])
            curve_np = create_roc_curve(all_labels, all_probs[:, 0])
            logger.log_image("val/roc_curve", curve_np)
        else:
            auc_pr = 0.1
            roc_auc = 0.5

        logger.log_dict({"val/balanced_acc": balanced_acc})
        logger.log_dict({"val/roc_auc": roc_auc})
        logger.log_dict({"val/mcc": mcc})
        logger.log_dict({"val/weighted_f1": weighted_f1})
        logger.log_dict({"val/macro_f1": macro_f1})
        logger.log_dict({"val/auc_pr": auc_pr})

        metrics = {
            "roc_auc": roc_auc,
            "balanced_acc": balanced_acc,
            "mcc": mcc,
            "auc_pr": auc_pr,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
        }
        return metrics

def train_k_fold(cfg: DictConfig):
    # Determine cohorts based on the save directory
    cohorts = ["BWHQuality", "NTU", "MRXS"]

    # Logger initialization
    logger = instantiate(cfg.logger)
    logger.log_cfg(OmegaConf.to_container(cfg, resolve=True))
    fisrt_save = False  # Flag to save the first model with sufficient performance

    use_mixed_precision = False
    scaler = GradScaler() if use_mixed_precision else None

    for fold in range(cfg.data.folds):
        print(f"fold is {fold}")
        logger.set_fold(fold, cfg)
        
        # Get train, validation, and test datasets for the fold
        train, val, test = get_datasets_for_fold(cfg, fold)
        print(f"get dataset done")
        sampler = WeightedRandomSampler(train.weights, len(train.weights), replacement=True)
        print("weighted sampler done")
        
        # Data loaders
        train_dl = DataLoader(train, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, sampler=sampler)
        val_dl = DataLoader(val, batch_size=cfg.data.batch_size, shuffle=False)
        test_dl = DataLoader(test, batch_size=cfg.data.batch_size, shuffle=False)
        print("dataloader done")
        
        # Build model and move it to the specified device
        model = build_E2E_pipeline(cfg).to(cfg.train.device)
        print("model built")
        
        # Optimizer: only trainable parameters
        optimizer = instantiate(cfg.optimizer, filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr)

        # Calculate positive weight for the criterion based on label distribution
        label_counts = {}
        for slide_id in train.slide_ids:
            label = train.get_label(slide_id)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Compute pos_weight
        num_pos = label_counts.get(1, 0)
        num_neg = label_counts.get(0, 0)
        if num_pos > 0:
            pos_weight = torch.FloatTensor([num_neg / num_pos]).to(cfg.train.device)
        else:
            pos_weight = torch.FloatTensor([1.0]).to(cfg.train.device)  # Fallback in case of no positive samples

        # Loss function with dynamically computed pos_weight
        criterion = nn.BCEWithLogitsLoss() # pos_weight=pos_weight I remove the pos_weight or too much emphasis on class 1.

        # Early stopping variables
        if cfg.train.early_stopping:
            best = 0
            patience = cfg.train.patience
            patience_counter = 0

        # Training loop
        for epoch in range(cfg.train.epochs):
            _ = train_one_epoch(cfg, model, train_dl, optimizer, criterion, cfg.train.device, epoch, logger)
            metrics = val_one_epoch(cfg, model, val_dl, criterion, cfg.train.device, epoch, logger)
            
            if ((fisrt_save==False) and (metrics["balanced_acc"] >= 0.92) and (metrics["auc_pr"] >= 0.92) and (metrics["weighted_f1"] >= 0.92) and (metrics["roc_auc"]>=0.92)):
                print("TRIGGERED PERFORMANT MODEL , SAVING...")
                save_model_clip(cfg,model,fold)
                fisrt_save = True
                
            if (fisrt_save==True):
                print(f"Good model found, stopping at : {epoch + 1} epochs.")
                break
            
            if cfg.train.early_stopping:
                new = ( metrics["balanced_acc"] + 2.5*metrics["auc_pr"] + metrics["roc_auc"] + metrics["weighted_f1"] ) / 5.5
                if new > best:
                    best = new
                    patience_counter = 0
                    save_model_clip(cfg, model, fold)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
                    
       # TESTING TO COMPLETE
        print("Using clipped for model weights.")
        model.load_state_dict(torch.load(f"{cfg.checkpoints.save_dir}/clipped/{cfg.logger.experiment_id}_SC_CLIPPEDSAVE_{fold}.pth"))

        print("Best model Weights successfully loaded!!")
        # Record Metrics:
        device = cfg.train.device
        all_labels = []
        all_preds = []
        all_probs = []
        bar = tqdm(test_dl, total=len(test_dl), desc=f"Test set {fold}")
        model.eval()
        print("Model set to eval, Running Inference:")

        with torch.no_grad():
            for batch in bar:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                if use_mixed_precision:
                    with autocast():
                        output_logits = model(images)
                        output_probs = torch.sigmoid(output_logits)
                        preds = (output_probs >= 0.5).long()
                else:
                    output_logits = model(images)
                    output_probs = torch.sigmoid(output_logits)
                    preds = (output_probs >= 0.5).long()

                # Metrics collection
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(output_probs.detach().cpu().numpy())

            # === epoch metrics ===
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)
            
            print(f"all labels are {all_labels}")
            print(f"all probs are {all_probs[:, 0]}")
            
            thresholds = np.arange(0.0, 1.0, 0.0001)
            best_threshold = 0.5
            best_f1 = 0

            for threshold in thresholds:
                preds = (all_probs >= threshold).astype(int)
                f1 = f1_score(all_labels, preds, average="weighted")

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            # compute metrics with best threshold:
            print(f"Best thresh for Internal Testing is {best_threshold}")
            final_preds = (all_probs >= best_threshold).astype(int)
            print(f"and all preds at best tresh are {final_preds}")

            balanced_acc = balanced_accuracy_score(all_labels, final_preds)
            mcc = matthews_corrcoef(all_labels, final_preds)
            weighted_f1 = f1_score(all_labels, final_preds, average="weighted")
            macro_f1 = f1_score(all_labels, final_preds, average="macro")

            if cfg.data.n_classes == 2 and len(np.unique(all_labels)) > 1:
                auc_pr = average_precision_score(all_labels, all_probs[:, 0])
                roc_auc = roc_auc_score(all_labels, all_probs[:, 0])
            else:
                auc_pr = 0.1
                roc_auc = 0.5
            
            test_metrics = {
                "roc_auc": round(roc_auc, 3),
                "balanced_acc": round(balanced_acc, 3),
                "mcc": round(mcc, 3),
                "auc_pr": round(auc_pr, 3),
                "weighted_f1": round(weighted_f1, 3),
                "macro_f1": round(macro_f1, 3),
            }
            
            print("Saving Metrics on test set...")
            print(f"metrics are : {test_metrics}")
            update_metric_file(model_name=f"{cfg.logger.experiment_id}_{fold}_SC_CLIPPED", cohort=cfg.data.cohort, metrics=test_metrics)
            print("Done!")
            
            
        print("Running inference on other cohorts :")
        for cohort in cohorts:
            if cohort == cfg.data.cohort:
                continue
            print(f"Cohort is {cohort}")
            dataset = CytologyDataset(
                label_file=f"/n/data2/hms/dbmi/kyu/lab/gul075/{cohort}_Labels_SC_{cfg.data.label_column}.xlsx",
                image_folder=f"/n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_{cohort}_SC_100x",
                label_column= cfg.data.label_column,
                tile_number=cfg.data.tile_number,
                transform=None,
                augment=cfg.data.augment,
            )
            print(f"nombre de sample dans le dataset est {(len(dataset))}")
            
            dl = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.data.batch_size, shuffle=False
            )
            print("Dataloader done")
                        
            # Record Metrics:
            
            all_labels = []
            all_preds = []
            all_probs = []
            bar = tqdm(dl, total=len(dl), desc=f"Test set {fold}")
            model.eval()
            print("Model set to eval, Running Inference:")

            with torch.no_grad():
                for batch in bar:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)

                    if use_mixed_precision:
                        with autocast():
                            output_logits = model(images)
                            output_probs = torch.sigmoid(output_logits)
                            preds = (output_probs >= 0.5).long()
                    else:
                        output_logits = model(images)
                        output_probs = torch.sigmoid(output_logits)
                        preds = (output_probs >= 0.5).long()

                    # Metrics collection
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(output_probs.detach().cpu().numpy())

                # === epoch metrics ===
                all_labels = np.concatenate(all_labels)
                all_preds = np.concatenate(all_preds)
                all_probs = np.concatenate(all_probs)
                
                print(f"all labels are {all_labels}")
                print(f"all probs are {all_probs[:, 0]}")
                
                thresholds = np.arange(0.0, 1.0, 0.0001)
                best_threshold = 0.5
                best_f1 = 0

                for threshold in thresholds:
                    preds = (all_probs >= threshold).astype(int)
                    f1 = f1_score(all_labels, preds, average="weighted")

                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

                # compute metrics with best threshold:
                print(f"Best thresh for External {cohort} Testing is {best_threshold}")
                final_preds = (all_probs >= best_threshold).astype(int)
                print(f"and all preds at best tresh are {final_preds}")
                
                balanced_acc = balanced_accuracy_score(all_labels, final_preds)
                mcc = matthews_corrcoef(all_labels, final_preds)
                weighted_f1 = f1_score(all_labels, final_preds, average="weighted")
                macro_f1 = f1_score(all_labels, final_preds, average="macro")

                if cfg.data.n_classes == 2 and len(np.unique(all_labels)) > 1:
                    auc_pr = average_precision_score(all_labels, all_probs[:, 0])
                    roc_auc = roc_auc_score(all_labels, all_probs[:, 0])
                else:
                    auc_pr = 0.1
                    roc_auc = 0.5
                
                inf_metrics = {
                    "roc_auc": round(roc_auc,3),
                    "balanced_acc": round(balanced_acc,3),
                    "mcc": round(mcc,3),
                    "auc_pr": round(auc_pr,3),
                    "weighted_f1": round(weighted_f1,3),
                    "macro_f1": round(macro_f1,3),
                }
                
                print("Saving Metrics...")
                print(f"Inference metrics are : {inf_metrics}")
                update_metric_file(model_name = f"{cfg.logger.experiment_id}_{fold}_SC_CLIPPED", cohort = cohort , metrics = inf_metrics)
                print("Done!")
        fisrt_save = False
    
def save_model_clip(cfg, model, fold):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = cfg.checkpoints.save_dir
    save_dir = os.path.join(base_dir, save_dir, "clipped")
    os.makedirs(save_dir, exist_ok=True)
    model_pth = f"{cfg.logger.experiment_id}_SC_CLIPPEDSAVE_{fold}.pth"
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, model_pth))
    model.to(cfg.train.device)
    
def update_metric_file(model_name, cohort, metrics, metricfile=metricfile):
    metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])

    if os.path.exists(metricfile):
        df = pd.read_csv(metricfile)
    else:
        df = pd.DataFrame(columns=['MODEL'] + [model_name])
        df['MODEL'] = cohorts  

    if model_name not in df.columns:
        df[model_name] = ""

    if cohort not in df['MODEL'].values:
        new_row = pd.DataFrame({'MODEL': [cohort], model_name: [metrics_str]})
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df.loc[df['MODEL'] == cohort, model_name] = metrics_str

    df.to_csv(metricfile, index=False)

@hydra.main(version_base=None, config_path=cfgs_pth, config_name=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    pipeline_name = f"{cfg.extractor._target_}_{cfg.head._target_}"
    cfg.logger.experiment_id = f"{cfg.logger.experiment_id}.{pipeline_name}"
    
    
    print(f"CONFIG EXPERIMENT ID IS : {cfg.logger.experiment_id }")
    train_k_fold(cfg)
    
    
if __name__ == "__main__":
    main()
