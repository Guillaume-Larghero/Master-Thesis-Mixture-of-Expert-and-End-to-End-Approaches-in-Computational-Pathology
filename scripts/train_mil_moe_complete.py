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
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil

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

'''cfgs_pth = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)'''

cfgs_pth = "/home/gul075/MOE_github/MOE/configs"
cohorts=["BWHQuality" ,"NTU","MRXS","TVGH","KVGH","FEMH"]
metricfile = "/n/data2/hms/dbmi/kyu/lab/gul075/MOE_NEW_Metrics.csv"



def check_args(cfg: DictConfig):
    if cfg.data.k_fold:
        folds_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            cfg.data.folds_base_dir,
        )
        assert os.path.exists(folds_base_dir), f"{folds_base_dir} does not exist."


def get_datasets_for_fold(
    cfg: DictConfig, fold: int
) -> Tuple[H5MultiExpertDatasetInMemory, H5MultiExpertDatasetInMemory, H5MultiExpertDatasetInMemory]:
    folds_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg.data.folds_base_dir,
    )
    
    train_csv = os.path.join(folds_dir, f"train_fold_{fold}.csv")
    test_csv = os.path.join(folds_dir, f"test_fold_{fold}.csv")
    
    train_df = pd.read_csv(train_csv)
    
    # Stratified split into train and validation
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.25, 
        stratify=train_df["label"], 
        random_state=cfg.seed
    )
    
    temp_dir = os.path.join(folds_dir, f"temp_splits_{cfg.logger.experiment_id}")
    os.makedirs(temp_dir, exist_ok=True)
    temp_train_csv = os.path.join(temp_dir, f"train_fold_{fold}_split.csv")
    temp_val_csv = os.path.join(temp_dir, f"val_fold_{fold}_split.csv")
    
    train_df.to_csv(temp_train_csv, index=False)
    val_df.to_csv(temp_val_csv, index=False)
    
    test_dataset = H5MultiExpertDatasetInMemory(
        csv_path=test_csv,
        feat_folder=cfg.data.feat_folder,
        n_patches=cfg.data.n_patches,
        magnification=cfg.data.magnification,
        wsi_type=cfg.data.wsi_type,
        experts=cfg.moe.selected_experts,
        random_selection=cfg.data.random_selection.val,
        get_metadata=False,
    )
    
    train_dataset = H5MultiExpertDatasetInMemory(
        csv_path=temp_train_csv,
        feat_folder=cfg.data.feat_folder,
        n_patches=cfg.data.n_patches,
        magnification=cfg.data.magnification,
        wsi_type=cfg.data.wsi_type,
        experts=cfg.moe.selected_experts,
        random_selection=cfg.data.random_selection.train,
        get_metadata=False,
    )
    
    val_dataset = H5MultiExpertDatasetInMemory(
        csv_path=temp_val_csv,
        feat_folder=cfg.data.feat_folder,
        n_patches=cfg.data.n_patches,
        magnification=cfg.data.magnification,
        wsi_type=cfg.data.wsi_type,
        experts=cfg.moe.selected_experts,
        random_selection=cfg.data.random_selection.val,
        get_metadata=False,
    )
    
    shutil.rmtree(temp_dir)

    return train_dataset, val_dataset, test_dataset


def train_one_epoch(
    cfg: DictConfig,
    model: nn.Module,
    dl: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    logger: MetricsLogger,
):
    all_labels = []
    all_preds = []
    all_probs = []
    all_weights = []
    bar = tqdm(dl, total=len(dl), desc=f"Train Epoch {epoch}")
    model.train()
    z_loss = ZLoss(scale=cfg.train.z_loss_scale)

    for batch in bar:
        slide_ids, features, masks, labels = batch
        for i in range(len(features)):
            features[i] = features[i].to(device)
        for i in range(len(masks)):
            masks[i] = masks[i].to(device)
        labels = labels.to(device)
        output_probs, weights, logits = model(features, masks)
        preds = torch.argmax(output_probs, dim=1)

        # === loss calculation ===
        optimizer.zero_grad()
        loss = criterion(output_probs, labels)
        lb_loss = load_balancing_loss(weights)  # load balancing loss
        z_loss_value = z_loss(logits)  # z-loss computed on the raw logits
        loss = loss + lb_loss * cfg.train.lambda_balance + z_loss_value
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # === metrics ===
        acc = (preds == labels).float().mean()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(output_probs.detach().cpu().numpy())
        all_weights.append(weights.detach().cpu().numpy())
        logger.log_dict({"train/loss": loss.item()})
        logger.log_dict({"train/lb_loss": lb_loss.item()})
        logger.log_dict({"train/z_loss": z_loss_value.item()})
        logger.log_dict({"train/acc": acc.item()})

    # === epoch metrics ===
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_weights = np.concatenate(all_weights)
    mean_weights = np.mean(all_weights, axis=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    expert_names = [f"Expert {i+1}" for i in range(len(mean_weights))]
    expert_util = create_expert_util(expert_names, mean_weights)
    expert_variance = compute_expert_variance(all_weights)
    
    if (cfg.data.n_classes == 2) and (len(np.unique(all_labels)) > 1):
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        auc_pr = average_precision_score(all_labels, all_probs[:, 1])
        curve_np = create_roc_curve(all_labels, all_probs[:, 1])
        logger.log_image("train/roc_curve", curve_np)
    else:
        auc_pr = 0.1
        roc_auc = 0.5
    logger.log_dict({"train/balanced_acc": balanced_acc})
    logger.log_dict({"train/roc_auc": roc_auc})
    logger.log_dict({"train/mcc": mcc})
    logger.log_dict({"train/weighted_f1": weighted_f1})
    logger.log_dict({"train/macro_f1": macro_f1})
    logger.log_image("train/expert_util", expert_util)
    logger.log_dict({"train/expert_variance": expert_variance})
    logger.log_dict({"train/auc_pr": auc_pr})
    
    metrics = {
        "roc_auc": roc_auc,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "auc_pr": auc_pr,
        "weights": all_weights,
        "expert_variance": expert_variance,
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
    all_labels = []
    all_preds = []
    all_probs = []
    all_weights = []
    bar = tqdm(dl, total=len(dl), desc=f"Val Epoch {epoch}")
    model.eval()
    z_loss = ZLoss(scale=cfg.train.z_loss_scale)

    with torch.no_grad():
        for batch in bar:
            slide_ids, features, masks, labels = batch
            for i in range(len(features)):
                features[i] = features[i].to(device)
            for i in range(len(masks)):
                masks[i] = masks[i].to(device)
            labels = labels.to(device)
            output_probs, weights, logits = model(features, masks)
            preds = torch.argmax(output_probs, dim=1)

            # === loss calculation ===
            loss = criterion(output_probs, labels)
            lb_loss = load_balancing_loss(weights)  # load balancing loss
            z_loss_value = z_loss(logits)  # z-loss computed on the raw logits
            loss = loss + lb_loss * cfg.train.lambda_balance + z_loss_value

            # === metrics ===
            acc = (preds == labels).float().mean()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(output_probs.detach().cpu().numpy())
            all_weights.append(weights.detach().cpu().numpy())
            logger.log_dict({"val/loss": loss.item()})
            logger.log_dict({"val/lb_loss": lb_loss.item()})
            logger.log_dict({"val/z_loss": z_loss_value.item()})
            logger.log_dict({"val/acc": acc.item()})

        # === epoch metrics ===
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_weights = np.concatenate(all_weights)
        mean_weights = np.mean(all_weights, axis=0)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        macro_f1 = f1_score(all_labels, all_preds, average="weighted")
        expert_names = [f"Expert {i+1}" for i in range(len(mean_weights))]
        expert_util = create_expert_util(expert_names, mean_weights)
        expert_variance = compute_expert_variance(all_weights)

        if (cfg.data.n_classes == 2) and (len(np.unique(all_labels)) > 1):
            auc_pr = average_precision_score(all_labels, all_probs[:, 1])
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            curve_np = create_roc_curve(all_labels, all_probs[:, 1])
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
        logger.log_dict({"val/expert_variance": expert_variance})
        logger.log_image("val/expert_util", expert_util)
        metrics = {
            "roc_auc": roc_auc,
            "balanced_acc": balanced_acc,
            "mcc": mcc,
            "auc_pr": auc_pr,
            "weights": all_weights,
            "expert_variance": expert_variance,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
        }
        return metrics


def train_k_fold(cfg: DictConfig):
    if ("NPM1" in str(cfg.checkpoints.save_dir)) or ("FLT3" in str(cfg.checkpoints.save_dir)):
        cohorts=["BWHQuality" ,"NTU","MRXS"]
    else:
        cohorts = ["BWHQuality" ,"NTU","MRXS","TVGH","KVGH","FEMH"]
    logger = instantiate(cfg.logger)
    logger.log_cfg(OmegaConf.to_container(cfg, resolve=True))
    fisrt_save = False # Use to save the first model weights with sufficient performances

    for fold in range(cfg.data.folds):
        logger.set_fold(fold, cfg)
        train, val, test = get_datasets_for_fold(cfg, fold)
        sampler = WeightedRandomSampler(
            train.weights, len(train.weights), replacement=True
        )
        train_dl = torch.utils.data.DataLoader(
            train,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sampler=sampler,
        )
        val_dl = torch.utils.data.DataLoader(
            val, batch_size=cfg.data.batch_size, shuffle=False
        )
        
        test_dl = torch.utils.data.DataLoader(
            test, batch_size=cfg.data.batch_size, shuffle=False
        )

        model = build_simple_mil_moe(cfg).to(cfg.train.device)
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        label_counts = {}
        for slide_id in train.slide_ids:
            label = train.get_label(slide_id)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        total_samples = sum(label_counts.values())
        num_classes = len(label_counts)
        weights = [
            total_samples / (num_classes * label_counts[i])
            for i in sorted(label_counts)
        ]
        class_weights = torch.FloatTensor(weights).to(cfg.train.device)
        criterion = instantiate(cfg.criterion, weight=class_weights)

        if cfg.train.early_stopping:
            best = 0
            patience = cfg.train.patience
            patience_counter = 0

        for epoch in range(cfg.train.epochs):
            _ = train_one_epoch(
                cfg,
                model,
                train_dl,
                optimizer,
                criterion,
                cfg.train.device,
                epoch,
                logger,
            )
            metrics = val_one_epoch(
                cfg,
                model,
                val_dl,
                criterion,
                cfg.train.device,
                epoch,
                logger,
            )
            
            if ((fisrt_save==False) and (metrics["balanced_acc"] >= 0.9) and (metrics["auc_pr"] >= 0.75) and (metrics["weighted_f1"] >= 0.9) and (metrics["roc_auc"]>=0.9)):
                print("TRIGGERED PERFORMANT MODEL , SAVING...")
                save_first_model(cfg,model,fold)
                fisrt_save = True
                

            if cfg.train.early_stopping:
                new = ( metrics["balanced_acc"] + 2.5*metrics["auc_pr"] + metrics["roc_auc"] + metrics["weighted_f1"] ) / 5.5
                if new > best:
                    best = new
                    patience_counter = 0
                    save_model(cfg, model, fold)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
        
        
        print("Using Last for model weights.")
        model.load_state_dict(torch.load(f"{cfg.checkpoints.save_dir}/last/{cfg.logger.experiment_id}_LASTSAVE_{fold}.pth"))
        
        print("Best model Weights sucesfully loaded!!")
        #Record Metrics:
        device = cfg.train.device
        all_labels = []
        all_preds = []
        all_probs = []
        bar = tqdm(test_dl, total=len(test_dl), desc=f"Test set {fold}")
        model.eval()
        print("Model set to eval, Running Inference:")

        with torch.no_grad():
            for batch in bar:
                slide_ids, features, masks, labels = batch
                for i in range(len(features)):
                    features[i] = features[i].to(device)
                for i in range(len(masks)):
                    masks[i] = masks[i].to(device)
                labels = labels.to(device)
                output_probs, weights, logits = model(features, masks)
                preds = torch.argmax(output_probs, dim=1)

                # === metrics ===
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(output_probs.detach().cpu().numpy())

            # === epoch metrics ===
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            mcc = matthews_corrcoef(all_labels, all_preds)
            weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
            macro_f1 = f1_score(all_labels, all_preds, average="weighted")
            auc_pr = average_precision_score(all_labels, all_probs[:, 1])
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            
            test_metrics = {
                "roc_auc": round(roc_auc,3),
                "balanced_acc": round(balanced_acc,3),
                "mcc": round(mcc,3),
                "auc_pr": round(auc_pr,3),
                "weighted_f1": round(weighted_f1,3),
                "macro_f1": round(macro_f1,3),
            }
            
            print("Saving Metrics on test set...")
            print(f"metrics are : {test_metrics}")
            update_metric_file(model_name = f"{cfg.logger.experiment_id}_{fold}_LAST", cohort = cfg.data.cohort , metrics = test_metrics)
            print("Done!")
            
            
        print("Running inference on other cohorts :")
        for cohort in cohorts:
            if cohort == cfg.data.cohort:
                continue
            print(f"Cohort is {cohort}")
            dataset = H5MultiExpertDatasetInMemory(
                csv_path=f"{cfg.infdatasetpath}/{cohort}.csv",
                feat_folder=f"{cfg.featfolder}/Cytology_Feature_{cohort}_40x_Norm", #TO CHANGE WHEN OTHER MODEL
                n_patches=cfg.data.n_patches,
                magnification=cfg.data.magnification,
                wsi_type=cfg.data.wsi_type,
                experts=cfg.moe.selected_experts,
                random_selection=cfg.data.random_selection.val,
                get_metadata=False,
            )
            
            dl = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.data.batch_size, shuffle=False
            )
            print("Dataloader done")
                        
            all_labels = []
            all_preds = []
            all_probs = []
            bar = tqdm(dl, total=len(dl), desc=f"Inference Cohort {cohort}")
            model.eval()
            print("Model set to eval, Running Inference:")

            with torch.no_grad():
                for batch in bar:
                    slide_ids, features, masks, labels = batch
                    for i in range(len(features)):
                        features[i] = features[i].to(device)
                    for i in range(len(masks)):
                        masks[i] = masks[i].to(device)
                    labels = labels.to(device)
                    output_probs, weights, logits = model(features, masks)
                    preds = torch.argmax(output_probs, dim=1)

                    # === metrics ===
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(output_probs.detach().cpu().numpy())

                # === epoch metrics ===
                all_labels = np.concatenate(all_labels)
                all_preds = np.concatenate(all_preds)
                all_probs = np.concatenate(all_probs)
                balanced_acc = balanced_accuracy_score(all_labels, all_preds)
                mcc = matthews_corrcoef(all_labels, all_preds)
                weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
                macro_f1 = f1_score(all_labels, all_preds, average="weighted")
                auc_pr = average_precision_score(all_labels, all_probs[:, 1])
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
                
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
                update_metric_file(model_name = f"{cfg.logger.experiment_id}_{fold}_LAST", cohort = cohort , metrics = inf_metrics)
                print("Done!")
                
        if fisrt_save == True : 
            print("First save exist, using it for model weights.")           
            model.load_state_dict(torch.load(f"{cfg.checkpoints.save_dir}/first/{cfg.logger.experiment_id}_FIRSTSAVE_{fold}.pth"))
            
            print("Best model Weights sucesfully loaded!!")
            #Record Metrics:
            device = cfg.train.device
            all_labels = []
            all_preds = []
            all_probs = []
            bar = tqdm(test_dl, total=len(test_dl), desc=f"Test set {fold}")
            model.eval()
            print("Model set to eval, Running Inference:")

            with torch.no_grad():
                for batch in bar:
                    slide_ids, features, masks, labels = batch
                    for i in range(len(features)):
                        features[i] = features[i].to(device)
                    for i in range(len(masks)):
                        masks[i] = masks[i].to(device)
                    labels = labels.to(device)
                    output_probs, weights, logits = model(features, masks)
                    preds = torch.argmax(output_probs, dim=1)

                    # === metrics ===
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(output_probs.detach().cpu().numpy())

                # === epoch metrics ===
                all_labels = np.concatenate(all_labels)
                all_preds = np.concatenate(all_preds)
                all_probs = np.concatenate(all_probs)
                balanced_acc = balanced_accuracy_score(all_labels, all_preds)
                mcc = matthews_corrcoef(all_labels, all_preds)
                weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
                macro_f1 = f1_score(all_labels, all_preds, average="weighted")
                auc_pr = average_precision_score(all_labels, all_probs[:, 1])
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
                
                test_metrics = {
                    "roc_auc": round(roc_auc,3),
                    "balanced_acc": round(balanced_acc,3),
                    "mcc": round(mcc,3),
                    "auc_pr": round(auc_pr,3),
                    "weighted_f1": round(weighted_f1,3),
                    "macro_f1": round(macro_f1,3),
                }
                
                print("Saving Metrics on test set...")
                print(f"metrics are : {test_metrics}")
                update_metric_file(model_name = f"{cfg.logger.experiment_id}_{fold}_FIRST", cohort = cfg.data.cohort , metrics = test_metrics)
                print("Done!")
                
                
            print("Running inference on other cohorts :")
            for cohort in cohorts:
                if cohort == cfg.data.cohort:
                    continue
                print(f"Cohort is {cohort}")
                dataset = H5MultiExpertDatasetInMemory(
                    csv_path=f"{cfg.infdatasetpath}/{cohort}.csv",
                    feat_folder=f"{cfg.featfolder}/Cytology_Feature_{cohort}_40x_Norm", #TO CHANGE WHEN OTHER MODEL
                    n_patches=cfg.data.n_patches,
                    magnification=cfg.data.magnification,
                    wsi_type=cfg.data.wsi_type,
                    experts=cfg.moe.selected_experts,
                    random_selection=cfg.data.random_selection.val,
                    get_metadata=False,
                )
                
                dl = torch.utils.data.DataLoader(
                    dataset, batch_size=cfg.data.batch_size, shuffle=False
                )
                print("Dataloader done")
                            
                all_labels = []
                all_preds = []
                all_probs = []
                bar = tqdm(dl, total=len(dl), desc=f"Inference Cohort {cohort}")
                model.eval()
                print("Model set to eval, Running Inference:")

                with torch.no_grad():
                    for batch in bar:
                        slide_ids, features, masks, labels = batch
                        for i in range(len(features)):
                            features[i] = features[i].to(device)
                        for i in range(len(masks)):
                            masks[i] = masks[i].to(device)
                        labels = labels.to(device)
                        output_probs, weights, logits = model(features, masks)
                        preds = torch.argmax(output_probs, dim=1)

                        # === metrics ===
                        all_labels.append(labels.cpu().numpy())
                        all_preds.append(preds.cpu().numpy())
                        all_probs.append(output_probs.detach().cpu().numpy())

                    # === epoch metrics ===
                    all_labels = np.concatenate(all_labels)
                    all_preds = np.concatenate(all_preds)
                    all_probs = np.concatenate(all_probs)
                    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
                    mcc = matthews_corrcoef(all_labels, all_preds)
                    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
                    macro_f1 = f1_score(all_labels, all_preds, average="weighted")
                    auc_pr = average_precision_score(all_labels, all_probs[:, 1])
                    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
                    
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
                    update_metric_file(model_name = f"{cfg.logger.experiment_id}_{fold}_FIRST", cohort = cohort , metrics = inf_metrics)
                    print("Done!")
            fisrt_save=False



def save_model(cfg, model, fold):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = cfg.checkpoints.save_dir
    save_dir = os.path.join(base_dir, save_dir, "last")
    os.makedirs(save_dir, exist_ok=True)
    model_pth = f"{cfg.logger.experiment_id}_LASTSAVE_{fold}.pth"
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, model_pth))
    model.to(cfg.train.device)

def save_first_model(cfg, model, fold):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = cfg.checkpoints.save_dir
    save_dir = os.path.join(base_dir, save_dir, "first")
    os.makedirs(save_dir, exist_ok=True)
    model_pth = f"{cfg.logger.experiment_id}_FIRSTSAVE_{fold}.pth"
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, model_pth))
    model.to(cfg.train.device)

def update_metric_file(model_name, cohort, metrics, metricfile = metricfile):
    metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    
    if os.path.exists(metricfile):
        df = pd.read_csv(metricfile)
    else:
        df = pd.DataFrame(columns=['MODEL'] + [model_name])
        df['MODEL'] = cohorts  

    if model_name not in df.columns:
        df[model_name] = ""

    df.loc[df['MODEL'] == cohort, model_name] = metrics_str
    df.to_csv(metricfile, index=False)


@hydra.main(version_base=None, config_path=cfgs_pth, config_name=None)
def main(cfg: DictConfig):
    cfg.moe.expert_heads = [
        {
            "_target_": "models.owkin.abmil.ABMIL",
            "in_features": cfg.EMBEDDING_SIZES[expert],
            "out_features": cfg.data.n_classes,
            "metadata_cols": 0,
        }
        for expert in cfg.moe.selected_experts
    ]

    # === update dynamic variables ===
    OmegaConf.resolve(cfg)  # interpolate dynamic variables
    cfg.moe.router.out_features = len(cfg.moe.selected_experts)
    cfg.moe.router.in_features = sum(
        [
            cfg.moe.expert_heads[i]["in_features"]
            for i in range(len(cfg.moe.selected_experts))
        ]
    )
    check_args(cfg)
    seed_everything(cfg.seed)

    expert_names = "_".join(cfg.moe.selected_experts)
    cfg.logger.experiment_id = f"{cfg.logger.experiment_id}.{expert_names}_{cfg.group}"
    
    
    print(f"CONFIG EXPERIMENT ID IS : {cfg.logger.experiment_id }")
    train_k_fold(cfg)
    
    
if __name__ == "__main__":
    main()