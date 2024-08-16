"""
# Patch Extraction Script for Whole Slide Images (WSIs)

Extract image patches from WSIs at multiple magnifications.
It offers various options to control the patching process and provides visualization of the extracted patches.

python create_tiles.py [OPTIONS]

Options

--slide_folder: Path to WSI folder (can contain nested subfolders)
--patch_folder: Path to save patches and metadata.
--output_size: Patch size (default: 224 pixels).
--tissue_threshold: Minimum tissue percentage (default: 80%).
--magnifications: Magnification levels (default: 40, 20, 10).
--keep_top_n: Keep only top N patches with highest tissue percentage.
--keep_random_n: Keep only a random selection of N patches.
--n_workers: Number of worker processes for parallel processing.
--n_parts: Split slides into parts for parallel processing.
--part: Process only this part of the slides.
--only_coords: Only extract coordinates to a <slide_id>.h5 file.
    If this option is set, the script will only extract the coordinates of the highest magnification patches.
--use_center_mask: Use a center mask to define the viable patch area.
    This can be useful for slides that have a lot of black marker annotations around the edges.
    E.g. in hematological slides.
--center_mask_height: The height of the center mask as a fraction of the image height (default 0.5).
--center_mask_width: The width of the center mask as a fraction of the image width (default 0.9).
--is_Cytology: Set this argument to True if you want to tile a Cytology dataset (blood smear samples). If set to True, relevance of patches will be 
               computed using a custom score function that takes into account the clear separation of cell nucleus, the presence of cells and the bluriness of the patch ; instead of the quantity of tissue on the patch.

## Examples

Extract 224x224 patches at 40x, 20x, and 10x with 75% minimum tissue:

```python
python patch_extraction.py --slide_folder /path/to/slides --patch_folder /path/to/patches --output_size 224 --tissue_threshold 75 --magnifications 40 20 10
```

Keep only the top 1000 patches with the highest tissue percentage:

```python
python patch_extraction.py --slide_folder /path/to/slides --patch_folder /path/to/patches --keep_top_n 1000 --output_size 224 --tissue_threshold 75 --magnifications 40 20 10
```

Extract only the coordinates to a h5 file:

```python
python patch_extraction.py --slide_folder /path/to/slides --patch_folder /path/to/patches --output_size 224 --tissue_threshold 75 --magnifications 40 --only_coords
```
"""

import argparse
import datetime
import glob
import heapq
import logging
import math
import os
import pprint
import shutil
import sys
import threading
import zipfile
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import PIL 
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

ImageDict = dict[int, PIL.Image.Image]

MAX_VISUALIZE_PATCHES = 5000
Slide = openslide.OpenSlide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose magnification level and patch parameters."
    )
    parser.add_argument(
        "--slide_folder",
        type=str,
        default="/n/scratch/users/c/che099",
        help="Root slides folder.",
    )
    parser.add_argument(
        "--patch_folder",
        type=str,
        default="/home/chris/dataset/ebrains/patches",
        help="Root patch folder.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=224,
        help="Output size of each patch (default 224)",
    )
    parser.add_argument(
        "--tissue_threshold",
        type=int,
        default=80,
        help="Minimum tissue percentage threshold to save patches (default 80)",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="+",
        default=[40, 20, 10],
        help="Magnifications to extract patches for (default 40 20 10)",
    )
    parser.add_argument(
        "--keep_top_n",
        type=int,
        default=None,
        help="Keep only the top N patches with the highest tissue percentage (default None, which keeps all)",
    )
    parser.add_argument(
        "--keep_random_n",
        type=int,
        default=None,
        help="Keep only a maximum of random N patches (default None, which keeps all)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to use for processing patches in parallel(default 1)",
    )
    parser.add_argument(
        "--n_parts",
        type=int,
        default=1,
        help="The number of parts to split the slides into (default 1)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="The part of the slides to process (default 0)",
    )
    parser.add_argument(
        "--only_coords",
        action="store_true",
        help="Only extract coordinates to a <slide_id>.h5 file.",
    )
    parser.add_argument(
        "--use_center_mask",
        action="store_true",
        help="Instead of using otsu thresholding, use a center mask to define the viable patch area.",
    )
    parser.add_argument(
        "--center_mask_height",
        type=float,
        default=0.5,
        help="The height of the center mask as a fraction of the image height (default 0.5).",
    )
    parser.add_argument(
        "--center_mask_width",
        type=float,
        default=0.5,
        help="The width of the center mask as a fraction of the image width (default 0.9).",
    )
    parser.add_argument(
        "--is_Cytology",
        type=bool,
        default=False,
        help="Set up to True if you are tiling a Cytology dataset. This will rank patches according to the patch_score function instead of using tissue percentage"
    )
    return parser.parse_args()


@dataclass
class PatchConfig:
    slide_folder: str
    patch_folder: str
    output_size: int = 224
    tissue_threshold: int = 80
    magnifications: List[int] = field(default_factory=lambda: [40, 20, 10])
    keep_top_n: Optional[int] = None
    keep_random_n: Optional[int] = None
    n_workers: int = 1
    n_parts: int = 1
    part: int = 0
    only_coords: bool = False
    use_center_mask: bool = False
    center_mask_height: float = 0.5
    center_mask_width : float = 0.5
    is_Cytology: bool = False


def setup_folders(args: PatchConfig):
    os.makedirs(args.patch_folder, exist_ok=True)


def store_available_coords(args: PatchConfig, slide_id, coords: np.ndarray):
    if type(coords) != np.ndarray:
        coords = np.array(coords, dtype=[("x", np.int32), ("y", np.int32)])

    with h5py.File(f"{args.patch_folder}/{slide_id}.h5", "w") as f:
        _ = f.create_dataset("coords", data=coords)


def visualize_patches(args: PatchConfig, slide_path: str, target_mag: int = 20):
    wsi = openslide.OpenSlide(slide_path)
    slide_id = get_slide_id(slide_path)
    thumbnail = get_thumbnail(wsi)
    df = pd.read_csv(f"{args.patch_folder}/{slide_id}.csv")
    coords = df[df["magnification"] == target_mag][["x", "y"]].values
    level_0_dim = wsi.dimensions
    level_0_mag = int(wsi.properties["openslide.objective-power"])
    scaling_factor = level_0_mag / target_mag
    adjusted_patch_size = int(args.output_size * scaling_factor)
    downsample = level_0_dim[0] / thumbnail.shape[1]
    upsample = 1 / downsample
    thumbnail_patch_size = int(adjusted_patch_size * upsample)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(thumbnail)
    linewidth = 2 if len(coords) < 100 else 0.5

    if len(coords) > 2000:
        print(
            f"Warning: Got {len(coords)} patches for visualization. This might take a while."
        )
    if len(coords) > MAX_VISUALIZE_PATCHES:
        print(
            f"Warning: Got more than {MAX_VISUALIZE_PATCHES} patches, will only visualize a subset of {MAX_VISUALIZE_PATCHES} patches!"
        )
        indices = np.random.choice(len(coords), MAX_VISUALIZE_PATCHES, replace=False)
        coords = np.array(coords)[indices]

    for x, y in coords:
        top_left_x = int(x * upsample)
        top_left_y = int(y * upsample)
        rect = Rectangle(
            (top_left_x, top_left_y),
            thumbnail_patch_size,
            thumbnail_patch_size,
            linewidth=linewidth,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.axis("off")
    output_path = f"{args.patch_folder}/{slide_id}.png"
    plt.savefig(output_path)
    plt.close()


def setup_logging(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    today_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"{args.patch_folder}/{today_date}.log"
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # redirect outputs to file logger
    class StreamToLogger:
        def __init__(self, logger, log_level):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ""

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


def load_slides(args) -> List[str]:
    ndpi_files_direct = glob.glob(f"{args.slide_folder}/*.ndpi")
    svs_files_direct = glob.glob(f"{args.slide_folder}/*.svs")
    mrxs_files_direct = glob.glob(f"{args.slide_folder}/*.mrxs")
    ndpi_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.ndpi", recursive=True)
    svs_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.svs", recursive=True)
    mrxs_files_subdirs = glob.glob(f"{args.slide_folder}/**/*.mrxs", recursive=True)
    all_files = (
        ndpi_files_direct
        + svs_files_direct
        + ndpi_files_subdirs
        + svs_files_subdirs
        + mrxs_files_direct
        + mrxs_files_subdirs
    )
    all_files = list(set(all_files))
    all_files = sorted(all_files)
    return all_files


def load_success_ids(args) -> Set[str]:
    success_txt = f"{args.patch_folder}/success.txt"
    success_ids = set()
    if os.path.exists(success_txt):
        with open(success_txt) as f:
            success_ids = {line.strip() for line in f}
    return success_ids


def clean_unfinished(args: PatchConfig, slide_id):
    if os.path.exists(f"{args.patch_folder}/{slide_id}"):
        shutil.rmtree(f"{args.patch_folder}/{slide_id}")
    if os.path.exists(f"{args.patch_folder}/{slide_id}.csv"):
        os.remove(f"{args.patch_folder}/{slide_id}.csv")
    if os.path.exists(f"{args.patch_folder}/{slide_id}.zip"):
        os.remove(f"{args.patch_folder}/{slide_id}.zip")
        
def compute_patch_score(tile : PIL.Image.Image) -> float:

    #Calculate optical_density_sum and bluriness (var of laplacian):
    image_array = np.array(tile)
    grayscale = np.dot(image_array[:, :, :3], [0.2989, 0.5870, 0.1140])
    grayscale_cv = np.array(grayscale, dtype=np.uint8)
    normalized_grayscale = grayscale / 255.0
    normalized_grayscale[normalized_grayscale == 0] = 0.01 # avoid 0 division
    
    # Calculate optical density:
    optical_density = -np.log10(normalized_grayscale)
    optical_density_sum = np.sum(optical_density)
    
    #Calculate bluriness:
    laplacian_var = cv2.Laplacian(grayscale_cv, cv2.CV_64F).var()
    
    #calculate color percentage
    image_arrayHSV = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
    # Define color ranges for Blue and Pink in HSV
    # These ranges may need to be adjusted to better fit your specific dataset
    color_ranges = {
        'blue': ((100, 150, 50), (140, 255, 255)),  # Adjusted range for Blue
        'pink': ((150, 100, 100), (180, 255, 255))  # Adjusted range for Pink
    }

    color_percentages = {}
    # Initialize and process each color range
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # Create a mask for the current range
        current_mask = cv2.inRange(image_arrayHSV, lower, upper)
        
        # Calculate the percentage of pixels within this range
        color_percentage = np.sum(current_mask > 0) / current_mask.size
        
        # Store the result in a dictionary
        color_percentages[color] = color_percentage
        
    
    #Calculate cell_separation_score:
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = [np.mean(cnt, axis=0) for cnt in contours]
    distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            distances.append(dist)

    if distances:
        average_distance = np.mean(distances)
    else:
        average_distance = 0
    
    cell_separation_score = average_distance / (image_array.shape[0] * image_array.shape[1])  # Normalized by tile size
    
    #Check if values of the patch are in a relevant range and score them:
    #print(f"Optical_density score is : {optical_density_sum} and cell sep score is {cell_separation_score} and laplacian var is {laplacian_var}")
    # Good patch conditions
    if (15000 <= optical_density_sum <= 32000 and #12K
        0.25 < color_percentages['blue'] < 0.90 and #70
        0.002 < cell_separation_score  and # < 0.0038 
        laplacian_var > 40):
        return optical_density_sum + 10e9 + random.randint(1, 5000)

    # Additional condition for lesser good but still acceptable patches
    elif (8000 <= optical_density_sum <= 30000 and
          0.01 < color_percentages['blue'] < 0.70 and
          cell_separation_score > 0.001 and
          laplacian_var > 40):
        return optical_density_sum + 10e7 + random.randint(1, 5000)
    
    # Bad patch condition
    elif (8000 <= optical_density_sum <= 30000 and
          0.0 < color_percentages['blue'] and
          cell_separation_score > 0.001 and
          laplacian_var > 40):
        return optical_density_sum + 10e5 + random.randint(1, 5000)
    else:
        return random.randint(1,5000)

        
class PatchesQueue:
    """
    A simple priority queue to keep track of the top N patches.
    Helps to avoid sorting the patches at the end and keep memory usage low.
    """

    def __init__(self, max_size, is_Cytology=False):
        self.max_size = max_size
        self.is_Cytology = is_Cytology
        self.heap = []
        self.lock = threading.Lock()

    def check_value(self, value: float) -> bool:
        with self.lock:
            if len(self.heap) < self.max_size:
                return True
            elif value > self.heap[0][0]:
                return True
            return False

    def try_add_patch_tuple(
        self, value, patch_info: tuple, patches: Dict[int, PIL.Image.Image]
    ):
        with self.lock:
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (value, patch_info, patches))
            elif value > self.heap[0][0]:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, (value, patch_info, patches))

    def check_percentage(self, tissue_percentage: float) -> bool:
        if self.is_Cytology:
            return False
        return self.check_value(tissue_percentage)

    def check_score(self, score: float) -> bool:
        if not self.is_Cytology:
            return False
        return self.check_value(score)
        
    def get_top_patches(self):
        with self.lock:
            return [heapq.heappop(self.heap) for _ in range(len(self.heap))]


def get_slide_id(slide_path: str) -> str:
    fname = os.path.basename(slide_path)
    # tcga files have 10 "-" separated parts
    is_tcga = len(fname.split("-")) == 10
    if is_tcga:
        return os.path.basename(slide_path).split(".")[1]
    # Check if the filename contains multiple dots in it, ex cytology slide contains date : 0081__-__02.14.20.ndpi
    if fname.count('.') > 1:
        return '.'.join(fname.split('.')[:-1])
    return fname.split('.')[0]


def calculate_tissue_percentage(
    patch: PIL.Image.Image,
    lower_hsv: np.ndarray = np.array([0.5 * 255, 0.2 * 255, 0.2 * 255]),
    upper_hsv: np.ndarray = np.array([1.0 * 255, 0.7 * 255, 1.0 * 255]),
    patch_size: int = 224,
) -> float:
    hsv_image = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2HSV)
    tissue_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    tissue_area = np.count_nonzero(tissue_mask)
    total_area = patch_size**2
    return (tissue_area / total_area) * 100


def get_thumbnail(wsi: Slide, downsample: int = 16) -> np.ndarray:
    full_size = wsi.dimensions
    img_rgb = np.array(
        wsi.get_thumbnail(
            (int(full_size[0] / downsample), int(full_size[1] / downsample))
        )
    )
    return img_rgb


def get_tissue_mask(args: PatchConfig, img_rgb: np.ndarray):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 11)
    _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = img_otsu.astype(np.uint8)

    if args.use_center_mask:
        # Create a mask with a central rectangle activated
        center_mask = np.zeros_like(tissue_mask)
        height, width = center_mask.shape
        rect_height = int(height * args.center_mask_height)
        rect_width = int(width * args.center_mask_width)
        start_x = (width - rect_width) // 2
        start_y = (height - rect_height) // 2
        center_mask[start_y : start_y + rect_height, start_x : start_x + rect_width] = 1
        tissue_mask = cv2.bitwise_and(tissue_mask, tissue_mask, mask=center_mask)

    return tissue_mask


def find_tissue_patches(args: PatchConfig, wsi: Slide) -> List[Tuple[int, int]]:
    target_mag = max(args.magnifications)
    patch_size = args.output_size
    tissue_threshold = args.tissue_threshold / 100
    thumbnail = get_thumbnail(wsi)
    tissue_mask = get_tissue_mask(args, thumbnail)

    level_0_dim = wsi.dimensions
    level_0_mag = int(wsi.properties["openslide.objective-power"])

    # compute the downsample from the tissue mask dimensions to level_0_dim
    downsample = level_0_dim[0] / tissue_mask.shape[1]

    # scaling factor (level_0_mag to target_mag)
    scaling_factor = level_0_mag / target_mag

    # Adjust the patch size based on the magnification difference
    adjusted_patch_size = int(patch_size * scaling_factor)

    # now we create a grid of patches with the cell size being the adjusted_patch_size
    valid_patches = []

    # Iterate over the full image at level 0 with steps of adjusted_patch_size
    for y in range(0, level_0_dim[1], adjusted_patch_size):
        for x in range(0, level_0_dim[0], adjusted_patch_size):
            # Map the coordinates to the tissue mask scale
            mask_x = int(x / downsample)
            mask_y = int(y / downsample)

            area_start_x = max(mask_x - 1, 0)
            area_end_x = min(mask_x + 2, tissue_mask.shape[1])
            area_start_y = max(mask_y - 1, 0)
            area_end_y = min(mask_y + 2, tissue_mask.shape[0])

            # check if average intensity is above the threshold
            mask_area = tissue_mask[area_start_y:area_end_y, area_start_x:area_end_x]
            if np.mean(mask_area) > 255 * tissue_threshold:
                valid_patches.append((x, y))

    return valid_patches


def extract_patches(
    slide_path: str,
    center_location: Tuple[int, int],
    output_size: int,
    target_mags: List[int] = [40, 20, 10],
    is_Cytology : bool = False,
) -> Tuple[Dict[int, PIL.Image.Image], Dict[int, float]]:
    if type(output_size) not in [tuple, list]:
        output_size = (output_size, output_size)

    slide = openslide.OpenSlide(slide_path)
    highest_mag = float(slide.properties["openslide.objective-power"])
    native_magnifications = {
        highest_mag / slide.level_downsamples[level]: level
        for level in range(slide.level_count)
    }
    patches = {}
    scores = {}
    for target_mag in target_mags:
        if target_mag in native_magnifications:
            level = native_magnifications[target_mag]
            downsample = slide.level_downsamples[level]
            # calculate the center location at the native resolution
            center_x = int(center_location[0] * slide.level_downsamples[0])
            center_y = int(center_location[1] * slide.level_downsamples[0])
            # fix the center location to the top-left corner of the patch
            location = (
                int(center_x - output_size[0] // 2 * downsample),
                int(center_y - output_size[1] // 2 * downsample),
            )
            patch = slide.read_region(location, level, output_size)
        else:
            nearest_higher_mag = max(
                [mag for mag in native_magnifications if mag > target_mag],
                default=highest_mag,
            )
            nearest_higher_level = native_magnifications[nearest_higher_mag]
            scale_factor = nearest_higher_mag / target_mag
            extract_size = (
                round(output_size[0] * scale_factor),
                round(output_size[1] * scale_factor),
            )
            # calculate the center location at the highest resolution
            center_x = int(center_location[0] * slide.level_downsamples[0])
            center_y = int(center_location[1] * slide.level_downsamples[0])
            new_location = (
                round(center_x - extract_size[0] / 2),
                round(center_y - extract_size[1] / 2),
            )
            patch = slide.read_region(new_location, nearest_higher_level, extract_size)
            patch = patch.resize(output_size, Image.LANCZOS)
        if is_Cytology : 
            patches[target_mag] = patch
            scores[target_mag] = compute_patch_score(patches[target_mag])
        else:
            patches[target_mag] = patch.convert("RGB")
            scores[target_mag] = calculate_tissue_percentage(patches[target_mag])
    return patches, scores


def save_patches(patch_folder: str, slide_id: str, patches: ImageDict, idx: int):
    zip_path = f"{patch_folder}/{slide_id}.zip"
    with zipfile.ZipFile(zip_path, "a") as zipf:
        for magnification, patch in patches.items():
            img_byte_arr = BytesIO()
            patch.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            img_filename = f"{idx}_{magnification}.png"
            zipf.writestr(img_filename, img_byte_arr)


def process_patch(
    slide_path: str, x: int, y: int, args
) -> Optional[Tuple[ImageDict, int, int, int, int, float]]:
    """
    Processes a patch and returns a tuple of the patch, its coordinates, and the tissue percentage or None if the patch should be discarded.
    """
    try:
        patches, scores = extract_patches(
            slide_path, (x, y), args.output_size, args.magnifications , args.is_Cytology
        )
        if args.is_Cytology:
            return patches,x,y,scores
        else :
            if (
                calculate_tissue_percentage(patches[max(args.magnifications)])
                < args.tissue_threshold
            ):
                return None
            return patches, x, y, scores
    except Exception as e:
        print(f"Failed to process patch ({x},{y}) at {slide_path}: {e}")
    return None


def save_patch_and_record_info(patch_folder, result, slide_id, idx, args):
    if result:
        patches, x, y, scores = result
        save_patches(patch_folder, slide_id, patches, idx)
        csv_path = f"{patch_folder}/{slide_id}.csv"
        with open(csv_path, "a") as csv_file:
            for magnification, _ in patches.items():
                score = scores[magnification]
                csv_file.write(
                    f"{slide_id},{idx},{x},{y},{score},{args.output_size},{magnification}\n"
                )
        return 1
    return 0


def process_slide_random_n(args: PatchConfig, slide_path: str, patch_coords: list):
    slide_id = get_slide_id(slide_path)
    csv_header = "slide_id,idx,x,y,score,patch_size,magnification\n"
    csv_path = f"{args.patch_folder}/{slide_id}.csv"
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as csv_file:
            csv_file.write(csv_header)

    # === select a random subset of keep_random_n patches ===
    total_patches = min(args.keep_random_n, len(patch_coords))
    indices = np.random.choice(len(patch_coords), total_patches, replace=False)
    patch_coords = np.array(patch_coords)[indices]
    # we already filtered the patches in the previous masking step
    args.tissue_threshold = 0.0

    patches_queue = PatchesQueue(args.keep_random_n)
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor, tqdm(
        total=total_patches, desc=f"Processing {slide_id}"
    ) as pbar:
        tasks = {
            executor.submit(process_patch, slide_path, x, y, args): (
                x,
                y,
            )
            for x, y in patch_coords
        }
        patch_idx = 0
        for future in as_completed(tasks):
            result = future.result()

            if result:
                # x,y = center_x, center_y
                patches, x, y, scores = result
                patch_info = (x, y, scores, patch_idx)
                patch_idx += 1
                # create dictionary for patches
                patches_dict = {
                    magnification: patches[magnification]
                    for magnification in args.magnifications
                }
                # add patches to the queue
                patches_queue.try_add_patch_tuple(
                    scores[max(args.magnifications)],
                    patch_info,
                    patches_dict,
                )
            pbar.update(1)

    # save the top N patches to the csv file and zip file
    top_patches = patches_queue.get_top_patches()

    for _, patch_info, patch_tuple in tqdm(
        top_patches, desc=f"Saving patches for {slide_id}"
    ):
        x, y, scores, idx = patch_info
        save_patches(args.patch_folder, slide_id, patch_tuple, idx)
        csv_path = f"{args.patch_folder}/{slide_id}.csv"
        with open(csv_path, "a") as csv_file:
            for magnification, _ in patch_tuple.items():
                csv_file.write(
                    f"{slide_id},{idx},{x},{y},{scores[magnification]},{args.output_size},{magnification}\n"
                )


def process_slide_top_n(args: PatchConfig, slide_path: str, patch_coords: list):
    print(f"Option: keep_top_n = {args.keep_top_n}")
    slide_id = get_slide_id(slide_path)
    csv_header = "slide_id,idx,x,y,score,patch_size,magnification\n"
    csv_path = f"{args.patch_folder}/{slide_id}.csv"
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as csv_file:
            csv_file.write(csv_header)

    total_patches = len(patch_coords)
    patches_queue = PatchesQueue(args.keep_top_n)
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor, tqdm(
        total=total_patches, desc=f"Processing {slide_id}"
    ) as pbar:
        tasks = {
            executor.submit(process_patch, slide_path, x, y, args): (
                x,
                y,
            )
            for x, y in patch_coords
        }
        patch_idx = 0
        for future in as_completed(tasks):
            result = future.result()

            if result:
                # x,y = center_x, center_y
                patches, x, y, scores = result
                patch_info = (x, y, scores, patch_idx)
                patch_idx += 1
                # create dictionary for patches
                patches_dict = {
                    magnification: patches[magnification]
                    for magnification in args.magnifications
                }
                # add patches to the queue
                patches_queue.try_add_patch_tuple(
                    scores[max(args.magnifications)],
                    patch_info,
                    patches_dict,
                )
            pbar.update(1)

    # save the top N patches to the csv file and zip file
    top_patches = patches_queue.get_top_patches()

    for _, patch_info, patch_tuple in tqdm(
        top_patches, desc=f"Saving patches for {slide_id}"
    ):
        x, y, scores, idx = patch_info
        save_patches(args.patch_folder, slide_id, patch_tuple, idx)
        csv_path = f"{args.patch_folder}/{slide_id}.csv"
        with open(csv_path, "a") as csv_file:
            for magnification, _ in patch_tuple.items():
                csv_file.write(
                    f"{slide_id},{idx},{x},{y},{scores[magnification]},{args.output_size},{magnification}\n"
                )


def process_slide(args: PatchConfig, slide_path: str, patch_coords: list):
    slide_id = get_slide_id(slide_path)

    # prepare csv
    csv_header = "slide_id,idx,x,y,score,patch_size,magnification\n"
    csv_path = f"{args.patch_folder}/{slide_id}.csv"
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as csv_file:
            csv_file.write(csv_header)

    total_patches = len(patch_coords)
    all_results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor, tqdm(
        total=total_patches
    ) as pbar:
        tasks = {
            executor.submit(process_patch, slide_path, x, y, args): (
                x,
                y,
            )
            for x, y in patch_coords
        }
        saved_patches = 0
        for future in as_completed(tasks):
            result = future.result()

            if result:
                all_results.append((result, saved_patches))  # (result, idx)

            saved_patches += save_patch_and_record_info(
                args.patch_folder,
                result,
                slide_id,
                saved_patches,
                args,
            )
            pbar.update(1)

    print(f"Saved {saved_patches} patches for slide id {slide_id}")


def keep_top_n(args, slide_id: str):
    """
    Keeps only the top k patches with the highest tissue percentage.
    """
    csv = pd.read_csv(f"{args.patch_folder}/{slide_id}.csv")

    # sort by tissue percentage, but only for 40x patches
    sorted_csv = csv[csv["magnification"] == 40]
    sorted_csv = sorted_csv.sort_values(by="score", ascending=False)
    keep_csv = sorted_csv.head(args.keep_top_n)
    keep_idx = set(keep_csv["idx"])
    for idx in tqdm(csv["idx"], desc=f"Cleaning up {slide_id}"):
        if idx not in keep_idx:
            for magnification in args.magnifications:
                patch_path = f"{args.patch_folder}/{slide_id}/{idx}_{magnification}.png"
                if os.path.exists(patch_path):
                    os.remove(patch_path)

    # write the new csv file that replaces the original one
    # but that contains all magnifications
    new_csv = csv[csv["idx"].isin(keep_idx)]
    new_csv.to_csv(f"{args.patch_folder}/{slide_id}.csv", index=False)


def clean_zip_files(args, slide_id: str):
    if args.keep_top_n is None:
        return
    # unzip the zip and remove the patches that are not in the top k
    zip_path = f"{args.patch_folder}/{slide_id}.zip"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(f"{args.patch_folder}/{slide_id}")
    os.remove(zip_path)
    # remove the patches that are not in the csv file
    csv = pd.read_csv(f"{args.patch_folder}/{slide_id}.csv")
    keep_idx = set(csv["idx"])
    magnifications = set(csv["magnification"])
    for idx in tqdm(
        os.listdir(f"{args.patch_folder}/{slide_id}"), desc=f"Cleaning up {slide_id}"
    ):
        if int(idx.split("_")[0]) not in keep_idx:
            os.remove(f"{args.patch_folder}/{slide_id}/{idx}")

    # create the new zip file and save all magnifications there
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for magnification in magnifications:
            for file in os.listdir(f"{args.patch_folder}/{slide_id}"):
                zipf.write(f"{args.patch_folder}/{slide_id}/{file}")

    # remove the folder with the patches
    shutil.rmtree(f"{args.patch_folder}/{slide_id}")


def main():
    args = parse_args()
    setup_folders(args)
    # setup_logging(args)

    print("=" * 50)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))
    print("=" * 50)

    # === Load all WSI files ===
    all_slides = load_slides(args)
    print(f"Loaded {len(all_slides)} slides.")

    success_ids = load_success_ids(args)
    print(f"Split slides into {args.n_parts} parts, processing part {args.part}")
    total_slides = len(all_slides)
    part_size = math.ceil(total_slides / args.n_parts)
    start_index = args.part * part_size
    end_index = min(start_index + part_size, total_slides)
    all_slides = all_slides[start_index:end_index]
    print(f"Process slide indices [{start_index}:{end_index}]")
    all_slides = [
        slide for slide in all_slides if get_slide_id(slide) not in success_ids
    ]
    print("Filtered out already previously processed slides.")

    for slide_path in all_slides:
        slide_id = get_slide_id(slide_path)
        if slide_id in success_ids:
            continue
        clean_unfinished(args, slide_id)

        try:
            print(f"Start processing slide: {slide_id}")
            wsi = openslide.OpenSlide(slide_path)
            patch_coords = find_tissue_patches(args, wsi)
            print(f"Found {len(patch_coords)} useful patches for slide {slide_id}")

            # store the available coords in a h5 file
            store_available_coords(args, slide_id, patch_coords)
            if not args.only_coords:
                if args.keep_top_n:
                    process_slide_top_n(args, slide_path, patch_coords)
                elif args.keep_random_n:
                    process_slide_random_n(args, slide_path, patch_coords)
                else:
                    process_slide(args, slide_path, patch_coords)

                visualize_patches(args, slide_path, target_mag=max(args.magnifications))

            with open(f"{args.patch_folder}/success.txt", "a") as f:
                f.write(f"{slide_id}\n")

        except Exception as e:
            print(f"Failed to process {slide_id}: {e}")
            with open("fails.txt", "a") as f:
                f.write(f"{slide_id}\n")


if __name__ == "__main__":
    main()
