"""
# Feature Extraction

Creates features based on the patches extracted using create_tiles.py.
The user can select a set of foundational models to extract features from the patches.
Currently, we support the following models:

- lunit
- resnet50
- uni
- swin224
- phikon
- ctrans
- chief
- plip
- gigapath
- cigar
"""

import argparse
import glob
import math
import os
import pprint
import shutil
import time
import zipfile
from functools import wraps
from typing import Dict, List, Set
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import sqlite3

from models.library import get_model, parse_model_type
from utils.transforms import get_transforms , get_transforms_SC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create features for patches using pre-defined models.",
        usage="""python create_features_from_patches.py
          --patch_folder PATH_TO_PATCHES
          --feat_folder PATH_TO_features
          [--models MODEL_TYPES]""",
    )
    parser.add_argument(
        "--patch_folder",
        type=str,
        help="Root patch folder. Patches are expected in <patch_folder>/<slide_id>.csv and <patch_folder>/<slide_id>/<idx>_<mag>.png.",
    )
    parser.add_argument(
        "--feat_folder",
        type=str,
        help="Root folder, under which the features will be stored: <feature_folder>/<slide_id>/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for processing the patches. Default: 1024.",
    )
    parser.add_argument(
        "--models",
        type=parse_model_type,
        default=[
            "ctrans",
            "lunit",
            "resnet50",
            "uni",
            "swin224",
            "phikon",
            "chief",
            "plip",
            "gigapath",
            "cigar",
        ],
        help="Comma-separated list of models to use (e.g., 'lunit,resnet50,uni,swin224,phikon').",
    )
    parser.add_argument(
        "--n_parts",
        type=int,
        default=1,
        help="The number of parts to split the items into (default 1)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="The part of the total items to process (default 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for processing (default 'cuda')",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="The huggingface token to use for downloading models (default None)",
    )
    return parser.parse_args()


def initialize_db(args):
    db_path = os.path.join(args.feat_folder, "success.db")
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS success (
                slide_id TEXT PRIMARY KEY,
                experts TEXT
            )
        """
        )
        conn.commit()
    finally:
        conn.close()


def load_success_data(args) -> pd.DataFrame:
    db_path = f"{args.feat_folder}/success.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM success", conn)
    conn.close()
    return df


def update_success(args, slide_id: str, models):
    db_path = f"{args.feat_folder}/success.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT experts FROM success WHERE slide_id = ?", (slide_id,))
    result = cursor.fetchone()
    if result:
        existing_experts = set(result[0].split(","))
        updated_experts = existing_experts.union(models)
        cursor.execute(
            "UPDATE success SET experts = ? WHERE slide_id = ?",
            (",".join(updated_experts), slide_id),
        )
    else:
        cursor.execute(
            "INSERT INTO success (slide_id, experts) VALUES (?, ?)",
            (slide_id, ",".join(models)),
        )
    conn.commit()
    conn.close()


def setup_folders(args):
    os.makedirs(args.feat_folder, exist_ok=True)
    assert os.path.exists(
        args.patch_folder
    ), f"Patch folder {args.patch_folder} does not exist."


def load_available_patches(args) -> Set[str]:
    available_patches_txt = f"{args.patch_folder}/success.txt"
    available_patch_ids = set()
    if os.path.exists(available_patches_txt):
        with open(available_patches_txt) as f:
            available_patch_ids = {line.strip() for line in f}
    return available_patch_ids


def load_all_csvs(args) -> List[str]:
    available_ids = load_available_patches(args)
    csvs = glob.glob(f"{args.patch_folder}/*.csv")
    csvs = sorted(csvs)
    csvs = [csv for csv in csvs if os.path.basename(csv)[:-4] in available_ids] 
    return csvs


def retry(max_retries=3, delay=5, exceptions=(Exception,)):
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = max_retries
            while retries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        raise
                    print(f"Retry {func.__name__} due to {e}, {retries} retries left.")
                    time.sleep(delay)

        return wrapper_retry

    return decorator_retry


@retry(max_retries=15, delay=5, exceptions=(OSError,))
def load_models(args) -> Dict[str, nn.Module]:
    models = {}
    for model in args.models:
        models[str(model)] = get_model(args, str(model)).to(args.device)
    return models


def unzip(args, slide_id: str):
    # unpacks the zip file
    os.makedirs(f"{args.feat_folder}/{slide_id}", exist_ok=True)
    zip_file = f"{args.patch_folder}/{slide_id}.zip"
    assert os.path.exists(zip_file), f"Zip file {zip_file} does not exist."
    print(f"Unpacking {zip_file}...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(f"{args.patch_folder}/{slide_id}")


def rmdir(args, slide_id: str):
    pth = f"{args.patch_folder}/{slide_id}"
    if os.path.exists(pth):
        shutil.rmtree(pth)


def load_patch(row, patch_folder: str, slide_id: str, transforms: nn.Module):
    idx, mag = row["idx"], row["magnification"]
    patch_png = f"{patch_folder}/{slide_id}/{idx}_{mag}.png"
    if os.path.exists(patch_png):
        patch = Image.open(patch_png).convert("RGB")
        return idx, mag, transforms(patch)
    return None, None, None


def get_features(
    batch: torch.Tensor, models: Dict[str, nn.Module]
) -> Dict[str, torch.Tensor]:
    batch_features = {model_type: [] for model_type in models.keys()}
    with torch.no_grad():
        for model_type, model in models.items():
            batch_features[model_type] = model(batch).detach().cpu()
    return batch_features


def store_metadata(args, slide_id: str):
    metadata = pd.read_csv(f"{args.patch_folder}/{slide_id}.csv")

    # drop column "uuid" if it exists
    if "slide_id" in metadata.columns:
        metadata.drop(columns=["slide_id"], inplace=True)

    for mag in metadata["magnification"].unique():
        filtered_metadata = metadata[metadata["magnification"] == mag]
        with h5py.File(
            f"{args.feat_folder}/{slide_id}/{mag}x_features.h5", "a"
        ) as h5_file:
            if "metadata" in h5_file.keys():
                del h5_file["metadata"]
            h5_file.create_dataset(
                "metadata",
                data=filtered_metadata.to_records(index=False),
                compression="gzip",
            )


def store_features(args, features_dict: Dict[str, torch.Tensor], slide_id: str):
    slide_dir = os.path.join(args.feat_folder, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    for model_type, mag_features in features_dict.items():
        for mag, features in mag_features.items():
            h5_file_path = os.path.join(slide_dir, f"{mag}x_features.h5")
            with h5py.File(h5_file_path, "a") as h5_file:  # Open in append mode
                model_name = str(model_type).upper()
                features_dataset_name = f"{model_name}_features"
                indices_dataset_name = f"{model_name}_indices"

                features_array = torch.stack(list(features.values())).numpy()
                indices_array = np.array(list(features.keys()), dtype="int")

                # Check if the dataset already exists
                if features_dataset_name in h5_file:
                    # If exists, replace the dataset
                    del h5_file[features_dataset_name]
                if indices_dataset_name in h5_file:
                    del h5_file[indices_dataset_name]

                # Create datasets for features and indices
                h5_file.create_dataset(
                    features_dataset_name,
                    data=features_array,
                    dtype="float32",
                    compression="gzip",
                )
                h5_file.create_dataset(
                    indices_dataset_name,
                    data=indices_array,
                    dtype="int",
                    compression="gzip",
                )


def process_folder(
    args,
    slide_id: str,
    models: Dict[str, nn.Module],
):
    print("processing_folder")
    csv_path = f"{args.patch_folder}/{slide_id}.csv"
    csv_data = pd.read_csv(csv_path)
    patch_info = csv_data[["idx", "magnification"]]
    features_dict = {str(m_type): {} for m_type in models}
    mini_batch = []
    batch_info = []

    total_rows = patch_info.shape[0]
    for _, row in tqdm(patch_info.iterrows(), total=total_rows, desc="Loading Patches"):
        idx, mag, patch = load_patch(row, args.patch_folder, slide_id, get_transforms_SC()) # change it if you re in SC or not
        if patch is None:
            print(f"Warning: Patch {idx}_{mag} not found! Skipping...")
            continue

        if patch is not None:
            mini_batch.append(patch)
            batch_info.append((idx, mag))

        if len(mini_batch) == args.batch_size:
            mini_batch = torch.stack(mini_batch).to(args.device)
            features = get_features(mini_batch, models)

            for model_type, features in features.items():
                for j, embedding in enumerate(features):
                    idx, mag = batch_info[j]
                    if mag not in features_dict[model_type]:
                        features_dict[model_type][mag] = {}
                    features_dict[model_type][mag][idx] = embedding

            # reset for the next batch
            mini_batch = []
            batch_info = []

    # handle remainders
    if mini_batch:
        mini_batch = torch.stack(mini_batch).to(args.device)
        features = get_features(mini_batch, models)
        for model_type, features in features.items():
            for j, embedding in enumerate(features):
                idx, mag = batch_info[j]
                if mag not in features_dict[model_type]:
                    features_dict[model_type][mag] = {}
                features_dict[model_type][mag][idx] = embedding

    store_features(args, features_dict, slide_id)
    store_metadata(args, slide_id)
    print("FEATURE STORED")


def main():
    args = parse_args()
    setup_folders(args)
    initialize_db(args)

    print("=" * 50)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))
    print("=" * 50)

    all_csvs = load_all_csvs(args)
    print(f"Found {len(all_csvs)} available slide bags to process.")

    print(f"Split items into {args.n_parts} parts, processing part {args.part}")
    total_csvs = len(all_csvs)
    part_size = math.ceil(total_csvs / args.n_parts)
    start_index = args.part * part_size
    end_index = min(start_index + part_size, total_csvs)
    all_csvs = all_csvs[start_index:end_index]
    slide_ids = [os.path.basename(csv)[:-4] for csv in all_csvs]
    print(f"Process slide indices [{start_index}:{end_index}]")

    models = load_models(args)

    success_data = load_success_data(args)
    requested_experts = set(str(m).upper() for m in args.models)

    for slide_id in slide_ids:
        if slide_id in success_data["slide_id"].values:
            processed_experts = set(
                success_data[success_data["slide_id"] == slide_id]["experts"]
                .str.split(",")
                .values[0]
            )
            if requested_experts.issubset(processed_experts):
                print(
                    f"Skipping already processed slide_id with all requested experts: {slide_id}"
                )
                continue
            else:
                print(f"Processing missing experts for slide_id: {slide_id}")

        try:
            unzip(args, slide_id)
        except Exception as e:
            print(f"Error unzipping {slide_id}: {e}")
            rmdir(args, slide_id)
            continue

        try:
            process_folder(args, slide_id, models)
        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
            rmdir(args, slide_id)
            continue

        update_success(args, slide_id, [m.upper() for m in models.keys()])
        rmdir(args, slide_id)


if __name__ == "__main__":
    main()
