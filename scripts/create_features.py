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

Currently deactivated:

- ctrans
"""

import argparse
import enum
import glob
import math
import os
import pprint
import shutil
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from utils.models import get_lunit, get_phikon, get_resnet50, get_swin_224, get_uni


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
    # TODO: add ctrans back here
    parser.add_argument(
        "--models",
        type=parse_model_type,
        default=["lunit", "resnet50", "uni", "swin224", "phikon"],
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


class ModelType(enum.Enum):
    CTRANS = 1
    LUNIT = 2
    RESNET50 = 3
    UNI = 4
    SWIN224 = 5
    PHIKON = 6
    NONE = None

    def __str__(self):
        return self.name


def parse_model_type(models_str):
    models = models_str.upper().split(",")
    try:
        return [ModelType[model] for model in models]
    except KeyError as e:
        raise argparse.ArgumentTypeError(f"Invalid model name: {e}")


def setup_folders(args):
    os.makedirs(args.feat_folder, exist_ok=True)
    assert os.path.exists(
        args.patch_folder
    ), f"Patch folder {args.patch_folder} does not exist."


def get_model(args, model: str) -> nn.Module:
    m_type = ModelType[model.upper()]
    if m_type == ModelType.LUNIT:
        model = get_lunit()
    # TODO: fix ctranspath
    # elif m_type == ModelType.CTRANS:
    #     model = get_ctrans()
    elif m_type == ModelType.RESNET50:
        model = get_resnet50()
    elif m_type == ModelType.UNI:
        model = get_uni(args.hf_token)
    elif m_type == ModelType.SWIN224:
        model = get_swin_224()
    elif m_type == ModelType.PHIKON:
        model = get_phikon()
    else:
        raise Exception("Invalid model type")
    return model


def load_success_ids(args) -> Set[str]:
    success_txt = f"{args.feat_folder}/success.txt"
    success_ids = set()
    if os.path.exists(success_txt):
        with open(success_txt) as f:
            success_ids = {line.strip() for line in f}
    return success_ids


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
    csvs = [csv for csv in csvs if os.path.basename(csv).split(".")[0] in available_ids]
    return csvs


def load_models(args) -> Dict[str, nn.Module]:
    models = {}
    for model in args.models:
        models[str(model)] = get_model(args, str(model))
    return models



def clean_unfinished(args, slide_id):
    if os.path.exists(f"{args.feat_folder}/{slide_id}"):
        shutil.rmtree(f"{args.feat_folder}/{slide_id}")


def get_transforms(
    mean: Optional[Tuple[int]] = (0.485, 0.456, 0.406),
    std: Optional[Tuple[int]] = (0.229, 0.224, 0.225),
) -> T.Compose:
    return T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


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


def load_patch(row, patch_folder, slide_id, transforms):
    idx, mag = row["idx"], row["magnification"]
    patch_png = f"{patch_folder}/{slide_id}/{idx}_{mag}.png"
    if os.path.exists(patch_png):
        patch = Image.open(patch_png).convert("RGB")
        return idx, mag, transforms(patch)
    return None, None, None


def get_features(
    batch: torch.Tensor, models: Dict[str, nn.Module]
) -> Dict[str, torch.Tensor]:
    batch_features = {}
    with torch.no_grad():
        for model_type, model in models.items():
            model = model.to(batch.device)
            output = model(batch).detach().cpu()
            batch_features[model_type] = output
    return batch_features


def store_features(args, features_dict: Dict[str, torch.Tensor], slide_id: str):
    # TODO: store idx_map more efficiently in csv with patch_idx: tensor_idx
    for model_type, mag_features in features_dict.items():
        for mag, features in mag_features.items():
            fname_np = os.path.join(
                args.feat_folder, slide_id, f"{mag}_{model_type}.npy"
            )
            fname_txt = os.path.join(
                args.feat_folder, slide_id, f"{mag}_{model_type}.txt"
            )

            all_tensors_np = torch.stack([v for _, v in features.items()]).numpy()
            idx_map = [k for k, _ in features.items()]

            # store tensors and idx_map
            with open(fname_np, "wb") as f:
                np.save(f, all_tensors_np)
            with open(fname_txt, "w") as f:
                f.write("\n".join(str(item) for item in idx_map))


def process_folder(
    args,
    slide_id: str,
    models: Dict[str, nn.Module],
):
    csv_path = f"{args.patch_folder}/{slide_id}.csv"
    csv_data = pd.read_csv(csv_path)
    patch_info = csv_data[["idx", "magnification"]]
    features_dict = {str(m_type): {} for m_type in models}
    mini_batch = []
    batch_info = []

    total_rows = patch_info.shape[0]
    for _, row in tqdm(patch_info.iterrows(), total=total_rows, desc="Loading Patches"):
        idx, mag, patch = load_patch(row, args.patch_folder, slide_id, get_transforms())
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
                    mag_key = f"{mag}x"
                    if mag_key not in features_dict[model_type]:
                        features_dict[model_type][mag_key] = {}
                    features_dict[model_type][mag_key][idx] = embedding

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
                mag_key = f"{mag}x"
                if mag_key not in features_dict[model_type]:
                    features_dict[model_type][mag_key] = {}
                features_dict[model_type][mag_key][idx] = embedding
    store_features(args, features_dict, slide_id)


def main():
    args = parse_args()
    setup_folders(args)

    print("=" * 50)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))
    print("=" * 50)

    all_csvs = load_all_csvs(args)
    print(f"Found {len(all_csvs)} available slide bags to process.")

    success_ids = load_success_ids(args)
    print(f"Split items into {args.n_parts} parts, processing part {args.part}")
    total_csvs = len(all_csvs)
    part_size = math.ceil(total_csvs / args.n_parts)
    start_index = args.part * part_size
    end_index = min(start_index + part_size, total_csvs)
    all_csvs = all_csvs[start_index:end_index]
    slide_ids = [os.path.basename(csv).split(".")[0] for csv in all_csvs]
    print(f"Process slide indices [{start_index}:{end_index}]")

    models = load_models(args)

    for slide_id in slide_ids:
        if slide_id in success_ids:
            continue

        unzip(args, slide_id)
        process_folder(args, slide_id, models)
        rmdir(args, slide_id)


if __name__ == "__main__":
    main()
