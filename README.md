# Master Thesis: Computational Pathology using Machine Learning on Whole-Slide Images

This repository contains the codebase for my Master's Thesis, focused on computational pathology using machine learning techniques applied to whole-slide images (WSIs). The goal is to predict leukemia subtypes and gene expression using two distinct pipelines: **Mixture of Experts (MOE)** and **End-to-End (E2E)**.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Pipeline 1: Mixture of Experts (MOE)](#pipeline-1-mixture-of-experts-moe)
- [Pipeline 2: End-to-End (E2E)](#pipeline-2-end-to-end-e2e)
- [Usage](#usage)
- [License](#license)

## Introduction

This project explores two different machine learning pipelines designed for digital pathology:
1. **Mixture of Experts (MOE)**: A pipeline that uses multiple foundation models as experts to extract features from WSI patches and combines their predictions through various strategies.
2. **End-to-End (E2E)**: A pipeline that extracts single-cell blast cells and trains a classifier directly on these cells without intermediate feature extraction steps.

## Installation

To run the code in this repository, you'll need to set up the required environment. Make sure to install all necessary dependencies by running:

```bash
pip install -r requirements.txt


## Pipeline 1: Mixture of Experts (MOE)

The **MOE pipeline** is designed to leverage multiple foundation models as experts, each providing predictions that are combined to make a final decision. The pipeline consists of the following components:

- **create_MOE_tiles** (located in `processing/`): Extracts useful patches at 40x magnification from WSIs.
- **Create_features** (located in `processing/`): Extracts feature vectors from the patches using foundation models.
- **normalize_patches** (located in `processing/`): Normalizes the extracted patches using Macenko normalization. This step is recommended to be run after `create_MOE_tiles` and before `Create_features`.
- **train_mil_moe_clipped.py** (located in `scripts/`): Trains, validates, and tests an MOE model. The model can be instantiated with a configurable number of experts. Each expert consists of a foundation model and a classification head. Available classification heads include ABMIL, DSMIL, MeanPool, TransMIL, and HiptMIL. The MOE strategy can be either `top-k`, which selects the best-performing model at inference, or `weighted sum`, which weights predictions according to the router's probability distribution.
- **finetuning_MOE** (located in `scripts/`): Fine-tunes a pre-trained MOE model to a different cohort.
