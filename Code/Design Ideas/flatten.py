import math
import os
import glob
import cv2 # Import OpenCV
import numpy as np # OpenCV uses NumPy arrays
import dlib # For face detection
from collections import defaultdict
import traceback

# --- Configuration ---
INPUT_FOLDER = "Design Ideas/INPUT" # Source images
OUTPUT_FOLDER = "Design Ideas/OUTPUT [FLATTENED]" # Output for this version
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" # Needed for dlib face detection

# Target size after resizing the face crop
TARGET_SIZE_W = 220
TARGET_SIZE_H = 220

# Padding around the detected face bounding box (percentage of width/height)
FACE_CROP_PADDING_FACTOR = 0.30 # e.g., 0.3 means 30% padding

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
# Ensure ranges are valid, e.g. OpenCV DEEP_RED from (248,255) becomes (174,179)
if OPENCV_HUE_CATEGORY_DEFINITIONS["DEEP_RED"][1] > 179:
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

# --- Helper Functions for Categorization (using OpenCV HSV ranges) ---
def get_hue_category_name_cv2(h_cv2, hue_definitions_cv2):
    if hue_definitions_cv2["DEEP_RED"][0] <= h_cv2 <= hue_definitions_cv2["DEEP_RED"][1]:
        return "DEEP_RED"
    for name, ranges in hue_definitions_cv2.items():
        if name == "DEEP_RED": continue
        if isinstance(ranges, tuple) and len(ranges) == 2:
            if ranges[0] <= h_cv2 <= ranges[1]:
                return name
    return "UNCLASSIFIED_HUE"

def get_saturation_category_name_cv2(s_cv2, saturation_bins):
    for name, (s_min, s_max) in saturation_bins.items():
        if s_min <= s_cv2 <= s_max:
            return name
    return "UNKNOWN_S"

def get_value_category_name_cv2(v_cv2, value_bins):
    for name, (v_min, v_max) in value_bins.items():
        if v_min <= v_cv2 <= v_max:
            return name
    return "UNKNOWN_V"

# --- Dlib Initialization ---
if not os.path.exists(PREDICTOR_PATH):
    print(f"ERROR: Dlib predictor file not found at '{PREDICTOR_PATH}' for face detection.")
    # Allow script to run if predictor isn't found, but face detection will fail gracefully.
    # For this script, only the face detector is strictly needed for bounding box.
    # If you intended to use landmarks from predictor later, this would be an exit()
    face_detector_dlib = dlib.get_frontal_face_detector() # Still useful
    landmark_predictor = None # Not used in this version for flat design
    print("WARNING: Landmark predictor not found. Face detection will provide bounding box only.")
else:
    face_detector_dlib = dlib.get_frontal_face_detector()
    # landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH) # Not strictly needed for this version


# --- Core Single Image Processing Function (using OpenCV and Dlib face crop) ---
def flatten_cropped_face_cv2(
    input_path,
    output_path,
    target_size=(TARGET_SIZE_W, TARGET_SIZE_H), # (width, height)
    blur_radius_approx=0.3,
    hue_definitions_cv2=OPENCV_HUE_CATEGORY_DEFINITIONS,
    saturation_bins=DEFAULT_SATURATION_BINS,
    value_bins=DEFAULT_VALUE_BINS,
    grayscale_thresholds=DEFAULT_GRAYSCALE_THRESHOLDS,
    padding_factor=FACE_CROP_PADDING_FACTOR
):
    category_data = defaultdict(lambda: {
        'sum_h_x': 0.0, 'sum_h_y': 0.0, 'sum_s': 0.0, 'sum_v': 0.0, 'count': 0, 'positions': []
    })

    try:
        img_bgr_full = cv2.imread(input_path)
        if img_bgr_full is None:
            print(f"Error: Could not read image at '{input_path}'. Skipping.")
            return False

        img_h_full, img_w_full = img_bgr_full.shape[:2]
        gray_image_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2GRAY)

        # 1. Detect face in the original full image
        faces_full = face_detector_dlib(gray_image_full)
        if not faces_full:
            print(f"  No faces detected in '{input_path}'. Skipping.")
            return "NO_FACE_ORIGINAL"
        
        face = faces_full[0] # Process first detected face
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        
        # 2. Calculate padded bounding box for the crop
        face_w, face_h = r - l, b - t
        pad_w, pad_h = int(face_w * padding_factor), int(face_h * padding_factor)
        crop_l, crop_t = max(0, l - pad_w), max(0, t - pad_h)
        crop_r, crop_b = min(img_w_full, r + pad_w), min(img_h_full, b + pad_h)

        # 3. Crop the face from the original image (BGR for HSV conversion)
        face_crop_orig_scale_bgr = img_bgr_full[crop_t:crop_b, crop_l:crop_r]
        if face_crop_orig_scale_bgr.size == 0:
            print(f"  Padded face crop for '{input_path}' resulted in an empty image. Skipping.")
            return False

        # 4. Resize this face crop to the TARGET_SIZE
        final_resized_face_bgr = cv2.resize(face_crop_orig_scale_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 5. Apply blur if specified (on the resized BGR face image)
        img_to_process_bgr = final_resized_face_bgr
        if blur_radius_approx > 0:
            k_val = int(blur_radius_approx * 5) 
            ksize_val = 2 * k_val + 1
            if ksize_val > 0: # ksize must be odd and positive
                 img_to_process_bgr = cv2.GaussianBlur(final_resized_face_bgr, (ksize_val, ksize_val), 0)
        
        # 6. Convert the processed (resized, possibly blurred) BGR face to HSV
        img_hsv = cv2.cvtColor(img_to_process_bgr, cv2.COLOR_BGR2HSV)
        height, width = img_hsv.shape[:2] # Should be target_size

        # --- Color Categorization Logic (same as before, but on the resized face crop) ---
        low_sat_thresh = grayscale_thresholds["VERY_LOW_SAT_THRESHOLD_FOR_GS"]
        black_thresh = grayscale_thresholds["VAL_THRESHOLD_BLACK_GS"]
        white_thresh = grayscale_thresholds["VAL_THRESHOLD_WHITE_GS"]

        for y_idx in range(height): # Renamed to avoid conflict with global y
            for x_idx in range(width): # Renamed to avoid conflict with global x
                h, s, v_val = img_hsv[y_idx, x_idx] # Renamed v to v_val
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
            if data['count'] == 0:
                average_hsv_for_category[cat_key] = (0, 0, 20)
                continue
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
                avg_h, avg_s, avg_v_val = average_hsv_for_category[cat_key] # Renamed avg_v
                avg_hsv_pixel_np = np.uint8([[[avg_h, avg_s, avg_v_val]]])
                avg_bgr_color_np = cv2.cvtColor(avg_hsv_pixel_np, cv2.COLOR_HSV2BGR)[0][0]
                for (px, py) in data['positions']:
                    output_img_bgr[py, px] = avg_bgr_color_np
        
        cv2.imwrite(output_path, output_img_bgr)
        return "SUCCESS"

    except Exception as e:
        print(f"An error occurred while processing '{input_path}': {e}")
        traceback.print_exc()
    return False

# --- Batch Processing Function ---
def batch_process_flat_faces(
    target_size=(TARGET_SIZE_W, TARGET_SIZE_H),
    blur_radius_approx=0.3,
    padding_factor=FACE_CROP_PADDING_FACTOR
):
    if face_detector_dlib is None :
        print("Dlib face detector not initialized. Cannot perform face cropping. Exiting batch process.")
        return

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found."); return
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER); print(f"Created output folder: '{OUTPUT_FOLDER}'")

    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_files = []
    for ext in image_extensions: image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    image_files = list(set(image_files))

    if not image_files: print(f"No supported image files found in '{INPUT_FOLDER}'."); return

    print(f"Found {len(image_files)} image files to process for flat face design.")
    processed_count = 0; failed_count = 0; no_face_orig_count = 0

    for i, input_image_path in enumerate(image_files):
        filename = os.path.basename(input_image_path)
        base, ext = os.path.splitext(filename)
        output_image_path = os.path.join(OUTPUT_FOLDER, f"{base}{ext}")

        print(f"Processing image {i+1}/{len(image_files)}: {filename} ... ", end="", flush=True)
        
        result = flatten_cropped_face_cv2(
            input_image_path, output_image_path, target_size, blur_radius_approx,
            padding_factor=padding_factor # Pass padding factor
        )
        
        if result == "SUCCESS":
            print("Done."); processed_count += 1
        elif result == "NO_FACE_ORIGINAL":
            print("No face detected."); no_face_orig_count +=1
        else: # False
            print("Failed."); failed_count += 1
            
    print(f"\nBatch processing complete.")
    print(f"Successfully processed: {processed_count} images.")
    print(f"Skipped (no face detected in original): {no_face_orig_count} images.")
    print(f"Failed to process (other errors): {failed_count} images.")

# --- Main Execution Block ---
if __name__ == '__main__':
    # Ensure your PREDICTOR_PATH is correct if you were to use landmarks,
    # but for now, only face_detector_dlib is used.
    if face_detector_dlib is None :
         print("Dlib face detector failed to initialize. Please check paths or Dlib installation.")
         exit()

    process_target_size = (TARGET_SIZE_W, TARGET_SIZE_H) # (width, height)
    process_blur_radius_approx = 0.0 # No blur by default for this flat style, adjust if needed
    process_face_padding = FACE_CROP_PADDING_FACTOR


    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found. Creating it with a dummy image.")
        os.makedirs(INPUT_FOLDER)
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw # Pillow for dummy
            temp_img = PILImage.new("RGB", (600, 800), (200, 220, 250))
            draw = PILImageDraw.Draw(temp_img)
            # Simulate a face-like structure for dlib to find
            face_color = (255, 200, 180)
            eye_color = (50, 50, 50)
            draw.ellipse((150, 150, 450, 650), fill=face_color) # Face oval
            draw.ellipse((220, 300, 280, 350), fill=eye_color)  # Left eye
            draw.ellipse((320, 300, 380, 350), fill=eye_color)  # Right eye
            draw.rectangle((270, 450, 330, 500), fill=eye_color) # Mouth
            temp_img.save(os.path.join(INPUT_FOLDER, "dummy_face_image.jpg"))
            print(f"Dummy image 'dummy_face_image.jpg' created in '{INPUT_FOLDER}'.")
        except Exception as e_dummy:
            print(f"Could not create dummy image: {e_dummy}.")


    print(f"Starting OpenCV batch processing for FLAT FACE designs...")
    print(f"Input folder: '{os.path.abspath(INPUT_FOLDER)}'")
    print(f"Output folder: '{os.path.abspath(OUTPUT_FOLDER)}'")
    print(f"Target Resized Face Size: {process_target_size[0]}x{process_target_size[1]}")
    print(f"Face Crop Padding: {process_face_padding*100}%")
    print(f"Blur Radius Approx: {process_blur_radius_approx}")


    batch_process_flat_faces(
        target_size=process_target_size,
        blur_radius_approx=process_blur_radius_approx,
        padding_factor=process_face_padding
    )