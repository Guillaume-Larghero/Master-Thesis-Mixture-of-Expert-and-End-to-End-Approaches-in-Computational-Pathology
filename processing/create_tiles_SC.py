"""
# Patch Extraction Script for Whole Slide Images (WSIs) at 100X (Single Cell Level)

python create_tiles_SC.py [OPTIONS]

Options

"input_dir": type=str, help=Path to the input directory containing .ndpi files.
"output_dir": type=str, help=Directory to save the extracted single-cell pictures and CSV files.
"""





import os
import cv2
import numpy as np
import csv
import shutil
from openslide import OpenSlide
from skimage import morphology
from skimage.morphology import h_maxima
from skimage.measure import label, regionprops
from PIL import Image




def apply_gamma_correction(image, gamma=1.5):
    """Applies gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def crop_and_save3(complete_image, centroid, shapeid, output_dir, csv_writer, slide_id, size=96, blur_threshold=100.0):
    """ Crops and saves a region from an image """
    nx_0 = max(int(centroid[0] - size / 2), 0)
    ny_0 = max(int(centroid[1] - size / 2), 0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    cropped_image = complete_image[ny_0:ny_1, nx_0:nx_1, :].astype(np.uint8)
    background_color = np.array([255, 255, 255], dtype=np.uint8)
    
    # Check the quality of the patch
    if is_qualitative(cropped_image) and (middle_white_area_size(cropped_image) > 850) and not is_blurry(cropped_image, blur_threshold):
        # Apply gamma correction
        gamma_corrected_image = apply_gamma_correction(cropped_image)
        b, g, r, a = cv2.split(gamma_corrected_image)
        alpha = a.astype(float) / 255
        blended_image = np.zeros((gamma_corrected_image.shape[0], gamma_corrected_image.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            blended_image[:, :, c] = (alpha * gamma_corrected_image[:, :, c] + (1 - alpha) * background_color[c]).astype(np.uint8)
                
        roi_file = os.path.join(output_dir, f'{shapeid}_40.png')
        cropped_image_pil = Image.fromarray(blended_image)  # Convert to PIL Image
        if cropped_image_pil.size == (96, 96):
            cropped_image_pil.save(roi_file)  # Save as RGBA PNG
            # Write to CSV
            csv_writer.writerow([slide_id, f'{shapeid}_40.png', 40])
            return True
    return False

def wbc_segmentation_hsv(img, outputdir=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Updated to COLOR_RGB2HSV
    gray = 255 - hsv[:, :, 1]
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    thresh_clean = 255 * morphology.remove_small_objects(thresh.astype(bool), min_size=2000, connectivity=4).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.erode(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    h_max = h_maxima(dist_transform, 1)
    h_max = cv2.dilate(h_max, kernel, iterations=3)
    ret, sure_fg = cv2.threshold(h_max, 0.3 * h_max.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), markers.astype(np.int32))
    bw = (markers > 1).astype(int)
    return 255 * bw.astype(np.uint8)

def chop_thumbnails(image, output_dir, csv_writer, slide_id, current_shapeid=0, max_cells_per_thumbnail=200, max_total_cells=2500):
    shapeid = current_shapeid
    image_np = np.array(image)  # Convert PIL Image to numpy array
    mp_masks = wbc_segmentation_hsv(image_np)
    output = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
    centroids = output[3]
    saved_cells = 0
    for c in centroids:
        if saved_cells >= max_cells_per_thumbnail or shapeid >= max_total_cells:
            break
        if crop_and_save3(image_np, c, shapeid, output_dir, csv_writer, slide_id):
            saved_cells += 1
            shapeid += 1
    return shapeid

def is_qualitative(patch, threshold=0.25):
    """ Checks if a patch has at least 20% background """
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    value_channel = hsv_patch[:, :, 2]
    background_mask = value_channel > 200  # Assume background is whitish if value is high
    background_ratio = np.sum(background_mask) / background_mask.size
    return background_ratio > threshold

def is_blurry(patch, threshold=100.0):
    """ Checks if a patch is too blurry using the Laplacian variance method """
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
    return laplacian_var < threshold

def middle_white_area_size(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    gray = 255 - hsv[:, :, 1]
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV) #127 before
    thresh_clean = 255 * morphology.remove_small_objects(thresh.astype(bool), min_size=2000, connectivity=4).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh_clean, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.erode(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    h_max = h_maxima(dist_transform, 1)
    h_max = cv2.dilate(h_max, kernel, iterations=3)
    ret, sure_fg = cv2.threshold(h_max, 0.3 * h_max.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB), markers.astype(np.int32))
    bw = (markers > 1).astype(np.uint8) * 255
    
    stack = [(48, 48)]
    count = 0

    # Directions for moving in the 4 connected neighbors (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Create a set to keep track of visited pixels
    visited = set()

    while stack:
        x, y = stack.pop()
        if (x, y) not in visited and np.array_equal(bw[y, x], 255):  # Check if pixel is white and not visited
            visited.add((x, y))
            count += 1
            for direction in directions:
                new_x, new_y = x + direction[0], y + direction[1]
                if 0 <= new_x < 96 and 0 <= new_y < 96:  # Check bounds
                    stack.append((new_x, new_y))
    
    return count


# Main script
def process_ndpi_images(input_dir, output_dir, roi_size=4096, max_cells_per_thumbnail=200, max_total_cells=2500):
    ndpi_files = [f for f in os.listdir(input_dir) if (f.endswith('.ndpi') or f.endswith('.mrxs'))]
    
    for ndpi_file in ndpi_files:
        base_name = os.path.splitext(ndpi_file)[0]
        csv_file_path = os.path.join(output_dir, f'{base_name}.csv')
        
        # Check if the file has already been processed
        if os.path.exists(csv_file_path):
            print(f"Skipping {ndpi_file}, already processed.")
            continue
        try:
            print(ndpi_file)
            input_file = os.path.join(input_dir, ndpi_file)
            
            # Open the .ndpi file
            slide = OpenSlide(input_file)
            
            # Get the dimensions of the whole slide image
            dimensions = slide.dimensions
            
            # Calculate the center point
            center_x, center_y = dimensions[0] // 2, dimensions[1] // 2
            
            # Determine the grid size to cover 2/3 of the height and 1/2 of the width
            grid_width = (dimensions[0] * 1/2) // roi_size
            grid_height = (dimensions[1] * 2/3) // roi_size
            
            shapeid = 0
            
            # Create output directory if it doesn't exist
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            slide_dir = os.path.join(output_dir, base_name)
            if not os.path.exists(slide_dir):
                os.makedirs(slide_dir)
            
            # Create CSV file
            csv_file_path = os.path.join(output_dir, f'{base_name}.csv')
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['slide_id', 'idx', 'magnification'])
                
                # Process each grid cell
                for i in range(int(grid_height)):
                    for j in range(int(grid_width)):
                        if shapeid >= max_total_cells:
                            break
                        top_left_x = int(center_x - (grid_width // 2) * roi_size + j * roi_size)
                        top_left_y = int(center_y - (grid_height // 2) * roi_size + i * roi_size)
                        
                        # Crop the ROI
                        roi_image = slide.read_region((top_left_x, top_left_y), 0, (roi_size, roi_size))
                        
                        # Convert to RGBA (if not already)
                        roi_image_rgba = roi_image.convert("RGBA")
                        roi_image_np = np.array(roi_image_rgba)
                        print(f"Processing thumbnail at ({top_left_x}, {top_left_y}) with shape {roi_image_np.shape}")
                        
                        # Extract single-cell pictures
                        shapeid = chop_thumbnails(roi_image_rgba, slide_dir, csv_writer, base_name, shapeid, max_cells_per_thumbnail, max_total_cells)
            
            # Create a ZIP file from the folder and delete the folder
            zip_file_path = os.path.join(output_dir, f'{base_name}.zip')
            shutil.make_archive(base_name, 'zip', slide_dir)
            shutil.move(f'{base_name}.zip', zip_file_path)
            shutil.rmtree(slide_dir)
        except Exception as e:
            print(f"Error processing {ndpi_file}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all .ndpi images in a directory and extract single-cell pictures.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing .ndpi files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the extracted single-cell pictures and CSV files.")
    
    args = parser.parse_args()
    
    process_ndpi_images(args.input_dir, args.output_dir)



# python slurm_features_SC.py /n/data2/hms/dbmi/kyu/lab/datasets/hematology/NTUHhematology/tissueImages/original/batch02 /n/data2/hms/dbmi/kyu/lab/gul075/Cytology_Tile_NTU_SC_100x
