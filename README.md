# Master Thesis: Mixture of Experts and End-to-End Approaches in Computational Pathology: Machine Learning for Leukemia Diagnosis and Genetic Analysis

This repository contains the codebase for my Master's Thesis, focused on computational pathology using machine learning techniques applied to whole-slide images (WSIs). The goal is to predict leukemia subtypes and gene expression using two distinct pipelines: **Mixture of Experts (MOE)** and **End-to-End (E2E)**.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Pipeline 1: Mixture of Experts (MOE)](#pipeline-1-mixture-of-experts-moe)
- [Pipeline 2: End-to-End (E2E)](#pipeline-2-end-to-end-e2e)
- [Usage](#usage)

## Introduction

This project explores two different machine learning pipelines designed for digital pathology:
1. **Mixture of Experts (MOE)**: A pipeline that uses multiple foundation models as experts to extract features from WSI patches and combines their predictions through various strategies.
2. **End-to-End (E2E)**: A pipeline that extracts single-cell blast cells and trains a classifier on feature vectors extracted by a convolutional Neural Network Backbone in an End-to-End fashion.

## Installation

To run the code in this repository, you'll need to set up the required environment. Make sure to install all necessary dependencies by running:

```bash
pip install -r requirements.txt
```



## Pipeline 1: Mixture of Experts (MOE)

The **MOE pipeline** is designed to leverage multiple foundation models as experts, each providing predictions that are combined to make a final decision. The pipeline consists of the following components:

- **create_MOE_tiles** (located in `processing/`): Extracts useful patches at 40x magnification from WSIs.
- **create_features** (located in `processing/`): Extracts feature vectors from the patches using foundation models. 10 foundation models are currently available with some needing an approved hugging face token or a wrapped .onnx file. See the models/library.py file for more details. So far the folloiwng mdoels are available : GigaPath , UNI , LUNIT , PHIKON , ResNet50 , CTransPath , Swin224 , CHIEF , CIGAR and PLIP.
- **normalize_patches** (located in `processing/`): Normalizes the extracted patches using Macenko normalization. This step is recommended to be run after `create_MOE_tiles` and before `Create_features`.
- **train_mil_moe_clipped.py** (located in `scripts/`): Trains, validates, and tests a MOE model. The model can be instantiated with a configurable number of experts. Each expert consists of a foundation model and a classification head. Available classification heads include ABMIL, DSMIL, MeanPool, TransMIL, and HiptMIL. The MOE strategy can be either `top-k`, which selects the best-performing model at inference, or `weighted sum`, which weights predictions according to the router's probability distribution.
- **finetuning_MOE** (located in `scripts/`): Fine-tunes a pre-trained MOE model to a different cohort.

## Pipeline 2: End-to-End (E2E)

The **E2E pipeline** is a more traditional and straightforward approach that focuses on single-cell analysis. The pipeline consists of the following components:

- **create_tiles_SC** (located in `processing/`): Extracts single-cell blast cells from WSIs at 100x magnification.
- **train_mil_e2e** (located in `scripts/`): Trains an end-to-end classifier. The classifier is instantiated with a feature extractor backbone and a classification head. The available classification heads include ABMIL, DSMIL, MeanPool, TransMIL, and HiptMIL.
- **finetune_E2E** (located in `scripts/`): Fine-tunes a trained E2E model to a desired cohort.

## Usage

Detailed instructions for running each scripts can be found on the scripts themselves. Below is a brief example of how to run the End to End pipeline:

1. **Extract Patches (Single cell - 100x)**:
   ```bash
   sbatch slurm_tiles_SC.sh /path/to/input/folders /path/to/output/folder

2. **Train End to End**:
   ```bash
    sbatch slurm_mil_e2e_GPU.sh CONFIG_NAME_TRAINING  #(without .yaml)

3. **Finetune on External Cohort**:
   ```bash
    sbatch slurm_finetuning_e2e.sh CONFIG_NAME_FINETUNING #(without .yaml)

   
