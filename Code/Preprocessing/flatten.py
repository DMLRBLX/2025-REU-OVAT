import math
import os
import glob
import cv2 # Import OpenCV
import numpy as np # OpenCV uses NumPy arrays
# import dlib # No longer needed for this version
from collections import defaultdict
import traceback

# --- Configuration ---
INPUT_FOLDER = "Code/INPUT [UTKFace]" # Source images PARENT folder
OUTPUT_FOLDER = "Code/design_ideas/OUTPUT [FLATTENED]" # Output PARENT folder
# PREDICTOR_PATH = "Code/face detector.dat" # No longer needed

# Target size for the resized image (full image will be resized to this)
TARGET_SIZE_W = 220
TARGET_SIZE_H = 220

# FACE_CROP_PADDING_FACTOR = 0.30 # No longer needed

# --- Color Category Bins and Thresholds (OpenCV Hue Scale 0-179) ---
_PILLOW_HUE_CATEGORY_DEFINITIONS = {
    "DEEP_RED": (248, 255), "RED": (0, 10), "RED_ORANGE": (11, 20),
    "ORANGE": (21, 35), "YELLOW_ORANGE": (36, 45), "YELLOW": (46, 58),
    "LIME_GREEN": (59, 75), "GREEN": (76, 110), "TEAL_CYAN": (111, 130),
    "CYAN": (131, 150), "AZURE_BLUE": (151, 170), "BLUE": (171, 190),
    "VIOLET": (191, 210), "PURPLE": (211, 225), "MAGENTA": (226, 238),
    "PINK_HUE": (239, 247),
}

def scale_hue_value_pillow_to_cv2(h_pillow):
    return min(179, max(0, round(h_pillow * 179.0 / 255.0)))

OPENCV_HUE_CATEGORY_DEFINITIONS = {
    name: (scale_hue_value_pillow_to_cv2(low), scale_hue_value_pillow_to_cv2(high if name != "DEEP_RED" else 255))
    for name, (low, high) in _PILLOW_HUE_CATEGORY_DEFINITIONS.items()
}
if OPENCV_HUE_CATEGORY_DEFINITIONS["DEEP_RED"][1] > 179: # Ensure max is 179
    OPENCV_HUE_CATEGORY_DEFINITIONS["DEEP_RED"] = (OPENCV_HUE_CATEGORY_DEFINITIONS["DEEP_RED"][0], 179)

DEFAULT_SATURATION_BINS = {
    "VERY_LOW_S": (0, 40), "LOW_S": (41, 80), "MEDIUM_S": (81, 150),
    "HIGH_S": (151, 210), "VERY_HIGH_S": (211, 255),
}
DEFAULT_VALUE_BINS = {
    "VERY_DARK_V": (0, 40), "DARK_V": (41, 80), "MEDIUM_V": (81, 150),
    "BRIGHT_V": (151, 210), "VERY_BRIGHT_V": (211, 255),
}
DEFAULT_GRAYSCALE_THRESHOLDS = {
    "VERY_LOW_SAT_THRESHOLD_FOR_GS": 20,
    "VAL_THRESHOLD_BLACK_GS": 60,
    "VAL_THRESHOLD_WHITE_GS": 200,
}

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')


# --- Helper Functions for Categorization (using OpenCV HSV ranges) ---
def get_hue_category_name_cv2(h_cv2, hue_definitions_cv2):
    # (Implementation is the same as your provided script)
    if hue_definitions_cv2["DEEP_RED"][0] <= h_cv2 <= hue_definitions_cv2["DEEP_RED"][1]:
        return "DEEP_RED"
    for name, ranges in hue_definitions_cv2.items():
        if name == "DEEP_RED": continue
        if isinstance(ranges, tuple) and len(ranges) == 2:
            if ranges[0] <= h_cv2 <= ranges[1]:
                return name
    return "UNCLASSIFIED_HUE"

def get_saturation_category_name_cv2(s_cv2, saturation_bins):
    # (Implementation is the same)
    for name, (s_min, s_max) in saturation_bins.items():
        if s_min <= s_cv2 <= s_max:
            return name
    return "UNKNOWN_S"

def get_value_category_name_cv2(v_cv2, value_bins):
    # (Implementation is the same)
    for name, (v_min, v_max) in value_bins.items():
        if v_min <= v_cv2 <= v_max:
            return name
    return "UNKNOWN_V"

# --- Dlib Initialization (No longer needed for face detection in this script) ---
# face_detector_dlib = None # Or remove entirely


# --- Core Single Image Processing Function ---
def flatten_image_cv2( # Renamed function
    input_path,
    output_path,
    target_size=(TARGET_SIZE_W, TARGET_SIZE_H), # (width, height)
    blur_radius_approx=0.3,
    hue_definitions_cv2=OPENCV_HUE_CATEGORY_DEFINITIONS,
    saturation_bins=DEFAULT_SATURATION_BINS,
    value_bins=DEFAULT_VALUE_BINS,
    grayscale_thresholds=DEFAULT_GRAYSCALE_THRESHOLDS
):
    category_data = defaultdict(lambda: {
        'sum_h_x': 0.0, 'sum_h_y': 0.0, 'sum_s': 0.0, 'sum_v': 0.0, 'count': 0, 'positions': []
    })

    try:
        img_bgr_full = cv2.imread(input_path)
        if img_bgr_full is None:
            print(f"Error: Could not read image at '{input_path}'.")
            return False

        # 1. Resize the ENTIRE image to the TARGET_SIZE
        # This might change the aspect ratio if original is different
        final_resized_image_bgr = cv2.resize(img_bgr_full, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. Apply blur if specified (on the resized BGR image)
        img_to_process_bgr = final_resized_image_bgr
        if blur_radius_approx > 0:
            k_val = int(blur_radius_approx * 5) 
            ksize_val = 2 * k_val + 1
            if ksize_val >= 3: 
                 img_to_process_bgr = cv2.GaussianBlur(final_resized_image_bgr, (ksize_val, ksize_val), 0)
        
        # 3. Convert the processed (resized, possibly blurred) BGR image to HSV
        img_hsv = cv2.cvtColor(img_to_process_bgr, cv2.COLOR_BGR2HSV)
        height, width = img_hsv.shape[:2] # Should be target_size dimensions

        # --- Color Categorization Logic ---
        low_sat_thresh = grayscale_thresholds["VERY_LOW_SAT_THRESHOLD_FOR_GS"]
        black_thresh = grayscale_thresholds["VAL_THRESHOLD_BLACK_GS"]
        white_thresh = grayscale_thresholds["VAL_THRESHOLD_WHITE_GS"]

        for y_idx in range(height):
            for x_idx in range(width):
                h, s, v_val = img_hsv[y_idx, x_idx]
                pos = (x_idx, y_idx)
                final_category_key = None

                if s < low_sat_thresh:
                    if v_val < black_thresh: final_category_key = "BLACK_GS"
                    elif v_val > white_thresh: final_category_key = "WHITE_GS"
                    else: final_category_key = "GRAY_GS"
                else:
                    h_cat_name = get_hue_category_name_cv2(h, hue_definitions_cv2)
                    s_cat_name = get_saturation_category_name_cv2(s, saturation_bins)
                    v_cat_name = get_value_category_name_cv2(v_val, value_bins)
                    
                    if h_cat_name != "UNCLASSIFIED_HUE" and \
                       s_cat_name != "UNKNOWN_S" and \
                       v_cat_name != "UNKNOWN_V":
                        final_category_key = (h_cat_name, s_cat_name, v_cat_name)
                    else: final_category_key = "GRAY_GS"

                data_entry = category_data[final_category_key]
                data_entry['positions'].append(pos)
                data_entry['count'] += 1
                data_entry['sum_s'] += float(s)
                data_entry['sum_v'] += float(v_val)

                if not isinstance(final_category_key, str) or "GS" not in final_category_key:
                    h_angle_rad = (float(h) / 179.0) * 2.0 * math.pi if 179.0 > 0 else 0
                    data_entry['sum_h_x'] += math.cos(h_angle_rad)
                    data_entry['sum_h_y'] += math.sin(h_angle_rad)
        
        average_hsv_for_category = {}
        for cat_key, data in category_data.items():
            if data['count'] == 0: average_hsv_for_category[cat_key] = (0,0,20); continue
            avg_s = int(round(data['sum_s'] / data['count']))
            avg_v = int(round(data['sum_v'] / data['count']))
            avg_h = 0
            if isinstance(cat_key, str) and "GS" in cat_key:
                avg_s = max(0, min(avg_s, low_sat_thresh -1 if low_sat_thresh > 0 else 0))
            elif isinstance(cat_key, tuple):
                if data['count'] > 0 and (data['sum_h_x'] != 0 or data['sum_h_y'] != 0):
                    mean_h_x = data['sum_h_x'] / data['count']
                    mean_h_y = data['sum_h_y'] / data['count']
                    avg_h_angle_rad = math.atan2(mean_h_y, mean_h_x)
                    if avg_h_angle_rad < 0: avg_h_angle_rad += 2.0 * math.pi
                    avg_h = min(179, max(0, int(round((avg_h_angle_rad / (2.0 * math.pi)) * 179.0))))
            average_hsv_for_category[cat_key] = (avg_h, avg_s, avg_v)

        output_img_bgr = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        for cat_key, data in category_data.items():
            if data['count'] > 0:
                avg_h, avg_s, avg_v_val = average_hsv_for_category[cat_key]
                avg_hsv_pixel_np = np.uint8([[[avg_h, avg_s, avg_v_val]]])
                avg_bgr_color_np = cv2.cvtColor(avg_hsv_pixel_np, cv2.COLOR_HSV2BGR)[0][0]
                for (px, py) in data['positions']:
                    output_img_bgr[py, px] = avg_bgr_color_np
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img_bgr)
        return True # Success

    except Exception as e:
        print(f"An error occurred while processing '{os.path.basename(input_path)}': {e}")
        traceback.print_exc()
    return False

# --- Batch Processing Function (Handles Nested Folders) ---
def batch_process_flatten_images_nested( # Renamed function
    input_parent_dir,
    output_parent_dir,
    target_size=(TARGET_SIZE_W, TARGET_SIZE_H),
    blur_radius_approx=0.3,
    hue_definitions=OPENCV_HUE_CATEGORY_DEFINITIONS,
    saturation_bins=DEFAULT_SATURATION_BINS,
    value_bins=DEFAULT_VALUE_BINS,
    grayscale_thresholds=DEFAULT_GRAYSCALE_THRESHOLDS
):
    if not os.path.isdir(input_parent_dir):
        print(f"Error: Input folder '{input_parent_dir}' not found.")
        return

    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)
        print(f"Created output folder: '{output_parent_dir}'")

    stats = {"SUCCESS": 0, "FAILED": 0, "TOTAL_FILES_SCANNED": 0} # Simplified stats
    
    print(f"\nScanning for images in '{input_parent_dir}' and its subfolders...")

    for root_dir, _, files in os.walk(input_parent_dir):
        files.sort() 
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                stats["TOTAL_FILES_SCANNED"] += 1
                input_image_path = os.path.join(root_dir, filename)
                
                relative_path_from_input_root = os.path.relpath(root_dir, input_parent_dir)
                output_sub_dir = os.path.join(output_parent_dir, relative_path_from_input_root)
                output_image_path = os.path.join(output_sub_dir, filename)

                print(f"Processing: {input_image_path} ... ", end="", flush=True)
                
                success = flatten_image_cv2( # Call renamed function
                    input_image_path, 
                    output_image_path, 
                    target_size, 
                    blur_radius_approx,
                    hue_definitions_cv2=hue_definitions,
                    saturation_bins=saturation_bins,
                    value_bins=value_bins,
                    grayscale_thresholds=grayscale_thresholds
                )
                
                if success:
                    print("Done.")
                    stats["SUCCESS"] += 1
                else:
                    print("Failed.")
                    stats["FAILED"] += 1
            
    print(f"\nBatch processing complete.")
    print(f"Total images scanned: {stats['TOTAL_FILES_SCANNED']}")
    print(f"Successfully processed: {stats['SUCCESS']} images.")
    print(f"Failed to process: {stats['FAILED']} images.")

# --- Main Execution Block ---
if __name__ == '__main__':
    current_input_folder = INPUT_FOLDER
    current_output_folder = OUTPUT_FOLDER

    process_target_size = (TARGET_SIZE_W, TARGET_SIZE_H)
    process_blur_radius_approx = 0.0 # No blur by default for this flat style

    # Dummy image creation (doesn't rely on face detection)
    if not os.path.exists(current_input_folder):
        print(f"Input folder '{current_input_folder}' not found. Creating it with a dummy image.")
        os.makedirs(current_input_folder, exist_ok=True)
        os.makedirs(os.path.join(current_input_folder, "SubfolderA"), exist_ok=True)
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw # Pillow for dummy
            temp_img = PILImage.new("RGB", (600, 400), (200, 220, 250)) # Dummy image
            draw = PILImageDraw.Draw(temp_img)
            draw.rectangle((50,50, 250,150), fill="red")
            draw.ellipse((300,50, 550,350), fill="blue")
            temp_img.save(os.path.join(current_input_folder, "SubfolderA", "dummy_color_image.jpg"))
            temp_img.save(os.path.join(current_input_folder, "dummy_root_color.png"))
            print(f"Dummy images created in '{current_input_folder}' and a subfolder.")
        except Exception as e_dummy:
            print(f"Could not create dummy image: {e_dummy}.")


    print(f"Starting OpenCV batch processing for FULL IMAGE FLATTENING (NO FACE CROP)...")
    print(f"Input folder: '{os.path.abspath(current_input_folder)}'")
    print(f"Output folder: '{os.path.abspath(current_output_folder)}'")
    print(f"Target Resized Image Size: {process_target_size[0]}x{process_target_size[1]}")
    print(f"Blur Radius Approx: {process_blur_radius_approx}")


    batch_process_flatten_images_nested( # Call renamed batch function
        current_input_folder,
        current_output_folder,
        target_size=process_target_size,
        blur_radius_approx=process_blur_radius_approx,
        hue_definitions=OPENCV_HUE_CATEGORY_DEFINITIONS,
        saturation_bins=DEFAULT_SATURATION_BINS,
        value_bins=DEFAULT_VALUE_BINS,
        grayscale_thresholds=DEFAULT_GRAYSCALE_THRESHOLDS
    )