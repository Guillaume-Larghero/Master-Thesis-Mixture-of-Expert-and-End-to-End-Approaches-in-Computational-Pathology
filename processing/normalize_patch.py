import os
import zipfile
import numpy as np
import cv2
import torch
from torchvision import transforms
import torchstain
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose paths and reference image for normalization"
    )
    parser.add_argument(
        "--patch_folder",
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/gul075/FEMH/107.8.6SCAN",
        help="Root slides folder.",
    )
    parser.add_argument(
        "--norm_patch_folder",
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/gul075/FEMH_single_cell",
        help="Root normalized slides folder.",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default="/n/data2/hms/dbmi/kyu/lab/gul075/Ref_Mackenko.png",
        help=".png to use as a reference for Mackenco normalization",
    )
    return parser.parse_args()

def mackenco_normalize(image, torch_normalizer, T):
    if image is None:
        print("Failed to load image.")
        return None

    t_image = T(image)
    try:
        norm, H, E = torch_normalizer.normalize(I=t_image, stains=True)
    except Exception as e:
        print(f"Normalization error: {e}")
        return None

    norm_image = norm.numpy().astype(np.uint8)
    return norm_image

def process_slide(slide_name, patch_folder, norm_patch_folder, torch_normalizer, T):
    zip_path = os.path.join(patch_folder, slide_name)
    print(f"zip_path is {zip_path}")
    norm_zip_path = os.path.join(norm_patch_folder, slide_name)
    print(f"norm_zip_path is {norm_zip_path}")
    
    if os.path.exists(norm_zip_path):
        print(f"Slide {slide_name} has already been processed. Skipping...")
        return

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(norm_patch_folder)
        extracted_files = zip_ref.namelist()
        print("zip file extracted")

    # Normalize the extracted .png files
    for file_name in extracted_files:
        if file_name.endswith('.png'):
            file_path = os.path.join(norm_patch_folder, file_name)
            image = cv2.imread(file_path)
            norm_image = mackenco_normalize(image, torch_normalizer, T)
            if norm_image is not None:
                cv2.imwrite(file_path, norm_image)
    print("png have been normalized")

    # Re-zip the normalized files
    with zipfile.ZipFile(norm_zip_path, 'w') as norm_zip_ref:
        for file_name in extracted_files:
            file_path = os.path.join(norm_patch_folder, file_name)
            norm_zip_ref.write(file_path, arcname=file_name)
            os.remove(file_path)
        print("png have been rezip")

def main():
    args = parse_args()
    print("argument parsed")
    
    if not os.path.exists(args.norm_patch_folder):
        print("creating norm patch folder")
        os.makedirs(args.norm_patch_folder)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    print("T setup")

    # Initialize the torchstain Macenko normalizer
    torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    print("torch_normalizer setup")

    # Load the target image and fit the normalizer
    reference_image = cv2.imread(args.reference_image)
    if reference_image is None:
        print(f"Failed to load reference image from {args.reference_image}")
        return
    torch_normalizer.fit(T(reference_image))
    print("ref image loaded")

    for slide_name in os.listdir(args.patch_folder):
        if slide_name.endswith('.zip'):
            print(f"slide name is : {slide_name}")
            process_slide(slide_name, args.patch_folder, args.norm_patch_folder, torch_normalizer, T)
            print(f"Processed slide: {slide_name}")

if __name__ == '__main__':
    main()
