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


cfgs_pth = "/home/gul075/MOE_github/MOE/configs"
cohorts=["BWHQuality","NTU","MRXS"]
featfolder = "/n/data2/hms/dbmi/kyu/lab/gul075"
metricfile = "/n/data2/hms/dbmi/kyu/lab/gul075/MOE_Metrics_FT_CLIPPED.csv"


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
def main(cfg : DictConfig):
    print("Main Starting...")
    
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
    
    
    cohorts = ["BWHQuality" ,"NTU","MRXS"]
    
    print("Seeding...")
    seed_everything(cfg.seed)
    device = cfg.train.device
    modelpath = cfg.modelpath
    modelname = cfg.modelname
    infdatasetpath = cfg.infdatasetpath
    test_size = cfg.test_size
    
    
    for fold in range(5):
        print(f"Starting fold {fold} : ")
        
        for cohort in cohorts:
            if cohort == "NTU":
                continue
            model = build_simple_mil_moe(cfg)
            print("Model instantiated.")
            weights_path = f'{modelpath}/{modelname}_{fold}.pth'
            print(f"weights_path is {weights_path}")
            model.load_state_dict(torch.load(weights_path))
            print("Weights sucesfully loaded!!")
            
            expert_names = "_".join(cfg.moe.selected_experts)
            cfg.logger.experiment_id = f"{modelname}_{expert_names}_FINETUNE_ON_{cohort}"
            logger = instantiate(cfg.logger)
            logger.log_cfg(OmegaConf.to_container(cfg, resolve=True))
            logger.set_fold(fold, cfg)
            
            print(f"Cohort is {cohort}")
            
            original_csv_path = f"{infdatasetpath}/{cohort}.csv"
            df = pd.read_csv(original_csv_path)
            stratify_col = 'label'  

            # Split into fine-tune and test datasets
            df_finetune, df_test = train_test_split(df, test_size=test_size, stratify=df[stratify_col] , random_state = cfg.seed)

            # Split the fine-tune dataset into fine-tune and validation datasets
            val_size = len(df_finetune) // 4
            df_finetune, df_val = train_test_split(df_finetune, test_size=val_size, stratify=df_finetune[stratify_col] , random_state = cfg.seed)

            # Save the splits data into temporary CSV files for the H5MultiExpertDataset
            finetune_csv_path = f"{infdatasetpath}/{cohort}_{cfg.logger.experiment_id}_FT_TEMP.csv"
            val_csv_path = f"{infdatasetpath}/{cohort}_{cfg.logger.experiment_id}_FTVAL_TEMP.csv"
            test_csv_path = f"{infdatasetpath}/{cohort}_{cfg.logger.experiment_id}_FTTEST_TEMP.csv"
            df_finetune.to_csv(finetune_csv_path, index=False)
            df_val.to_csv(val_csv_path, index=False)
            df_test.to_csv(test_csv_path, index=False)

            # Create fine-tune dataset
            finetune_dataset = H5MultiExpertDatasetInMemory(
                csv_path=finetune_csv_path,
                feat_folder=f"{featfolder}/Cytology_Feature_{cohort}_40x_Norm",
                n_patches=cfg.data.n_patches,
                magnification=cfg.data.magnification,
                wsi_type=cfg.data.wsi_type,
                experts=cfg.moe.selected_experts,
                random_selection=cfg.data.random_selection.val,
                get_metadata=False,
            )

            # Create sampler for the fine-tune dataset
            sampler = WeightedRandomSampler(
                finetune_dataset.weights, len(finetune_dataset.weights), replacement=True
            )

            # Create validation dataset
            val_dataset = H5MultiExpertDatasetInMemory(
                csv_path=val_csv_path,
                feat_folder=f"{featfolder}/Cytology_Feature_{cohort}_40x_Norm",
                n_patches=cfg.data.n_patches,
                magnification=cfg.data.magnification,
                wsi_type=cfg.data.wsi_type,
                experts=cfg.moe.selected_experts,
                random_selection=cfg.data.random_selection.val,
                get_metadata=False,
            )

            # Create test dataset
            test_dataset = H5MultiExpertDatasetInMemory(
                csv_path=test_csv_path,
                feat_folder=f"{featfolder}/Cytology_Feature_{cohort}_40x_Norm",
                n_patches=cfg.data.n_patches,
                magnification=cfg.data.magnification,
                wsi_type=cfg.data.wsi_type,
                experts=cfg.moe.selected_experts,
                random_selection=cfg.data.random_selection.val,
                get_metadata=False,
            )

            # Create data loaders
            finetune_dl = torch.utils.data.DataLoader(
                finetune_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, sampler=sampler,
            )
            
            val_dl = torch.utils.data.DataLoader(
                val_dataset, batch_size=cfg.data.batch_size, shuffle=False
            )
            
            test_dl = torch.utils.data.DataLoader(
                test_dataset, batch_size=cfg.data.batch_size, shuffle=False
            )
            
            optimizer = instantiate(cfg.optimizer, params=model.parameters())
            label_counts = {}
            for slide_id in finetune_dataset.slide_ids:
                label = finetune_dataset.get_label(slide_id)
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
                    finetune_dl,
                    optimizer,
                    criterion,
                    cfg.train.device,
                    epoch,
                    logger,
                )
                
                val_metrics = val_one_epoch(
                    cfg,
                    model,
                    val_dl,
                    criterion,
                    cfg.train.device,
                    epoch,
                    logger,
                )
                
                if cfg.train.early_stopping:
                    new = ( val_metrics["balanced_acc"] + 3.5*val_metrics["auc_pr"] + val_metrics["roc_auc"] + val_metrics["weighted_f1"] ) / 6.5
                    if new > best:
                        best = new
                        patience_counter = 0
                        save_model_path = f"{infdatasetpath}/{cohort}_{cfg.logger.experiment_id}_BESTMODEL_TEMP_FT.pth"
                        torch.save(model.cpu().state_dict(), save_model_path)
                        model.to(cfg.train.device)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs.")
                            break
                        
            #Loading best performing model :
            model.load_state_dict(torch.load(f"{infdatasetpath}/{cohort}_{cfg.logger.experiment_id}_BESTMODEL_TEMP_FT.pth"))
            print("Finetuned Weights sucesfully loaded!!")
            
            #Record Metrics after finetuning:
            all_labels = []
            all_preds = []
            all_probs = []
            bar = tqdm(test_dl, total=len(test_dl), desc=f"Inference Cohort {cohort}")
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
                
                metrics = {
                    "roc_auc": round(roc_auc,3),
                    "balanced_acc": round(balanced_acc,3),
                    "mcc": round(mcc,3),
                    "auc_pr": round(auc_pr,3),
                    "weighted_f1": round(weighted_f1,3),
                    "macro_f1": round(macro_f1,3),
                }
                
                print("Saving Metrics...")
                print(f"metrics are : {metrics}")
                update_metric_file(model_name = f"{modelname}_{fold}_FINETUNED{cfg.train.epochs}_seed{cfg.seed}", cohort = cohort , metrics = metrics)
                
                print("Removing Temp .csv  and Temp .pth...")
                os.remove(finetune_csv_path)
                os.remove(test_csv_path)
                os.remove(val_csv_path)
                os.remove(f"{infdatasetpath}/{cohort}_{cfg.logger.experiment_id}_BESTMODEL_TEMP_FT.pth")
                
                print("Done!")
                
                
                
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

    if cfg.data.n_classes == 2:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        auc_pr = average_precision_score(all_labels, all_probs[:, 1])
        curve_np = create_roc_curve(all_labels, all_probs[:, 1])
        logger.log_image("train/roc_curve", curve_np)
    else:
        auc_pr = None
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")
    logger.log_dict({"train/balanced_acc": balanced_acc})
    logger.log_dict({"train/roc_auc": roc_auc})
    logger.log_dict({"train/mcc": mcc})
    logger.log_dict({"train/weighted_f1": weighted_f1})
    logger.log_dict({"train/macro_f1": macro_f1})
    logger.log_image("train/expert_util", expert_util)
    logger.log_dict({"train/expert_variance": expert_variance})

    if auc_pr:
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

        if cfg.data.n_classes == 2:
            auc_pr = average_precision_score(all_labels, all_probs[:, 1])
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            curve_np = create_roc_curve(all_labels, all_probs[:, 1])
            logger.log_image("val/roc_curve", curve_np)
        else:
            auc_pr = None
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")

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
            
            
    
    
if __name__ == "__main__":
    main()