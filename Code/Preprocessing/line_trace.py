import cv2
import dlib
import numpy as np
import os
import math
import traceback # For more detailed error printing

# --- Configuration ---
INPUT_FOLDER = "Code/Input/input_fixed/INPUT [UTKFace]" # Source images
OUTPUT_FOLDER = "Code/preprocessing_output/utkface_output/OUTPUT [LT]"
PREDICTOR_PATH = "Code/face_detector.dat"

TARGET_SIZE_W = 220
TARGET_SIZE_H = 220 
FACE_CROP_PADDING_FACTOR = 0.30

DLIB_LINE_SETS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nostrils": list(range(31, 36)),
    "right_eye": list(range(36, 42)) + [36],
    "left_eye": list(range(42, 48)) + [42],
    "outer_lip": list(range(48, 60)) + [48],
    "inner_lip": list(range(60, 68)) + [60]
}
LANDMARK_LINE_COLOR = (0, 0, 0)
LANDMARK_LINE_THICKNESS = 1

APPLY_HSV_VALUE_NORMALIZATION = True
HSV_CONSTANT_V_VALUE = 200

APPLY_CLAHE_HOUGH = True # Used if APPLY_HSV_VALUE_NORMALIZATION is False
CLAHE_CLIP_LIMIT_HOUGH = 1.5
CLAHE_TILE_GRID_SIZE_HOUGH = (8,8)

APPLY_GAUSSIAN_BLUR_HOUGH = True # Used if APPLY_HSV_VALUE_NORMALIZATION is False
BLUR_KERNEL_HOUGH = (3,3)

APPLY_BILATERAL_FILTER = True # Used if APPLY_HSV_VALUE_NORMALIZATION is False
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

CANNY_THRESHOLD_1_HOUGH = 50
CANNY_THRESHOLD_2_HOUGH = 150

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 5
HOUGH_MIN_LINE_LENGTH = 10
HOUGH_MAX_LINE_GAP = 5

HOUGH_LINE_COLOR = (0, 0, 0)
HOUGH_LINE_THICKNESS = 1

SAVE_DEBUG_IMAGES = True
DEBUG_DIRS = {} # Will be populated in main setup

# --- Initialization of Dlib models ---
if not os.path.exists(PREDICTOR_PATH):
    print(f"ERROR: Dlib predictor file not found at '{PREDICTOR_PATH}'")
    # Consider exiting or raising an exception if critical
    face_detector_dlib = None
    landmark_predictor = None
else:
    face_detector_dlib = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

# --- Single Image Processing Function ---
def generate_single_image_trace(input_image_path, output_image_path, base_filename="image"):
    """
    Processes a single image to generate a face trace.
    Returns True on success, False on failure.
    """
    if face_detector_dlib is None or landmark_predictor is None:
        print("  Dlib models not loaded. Cannot process.")
        return False

    try:
        print(f"  Starting processing for: {os.path.basename(input_image_path)}")
        image_bgr_full = cv2.imread(input_image_path)
        if image_bgr_full is None:
            print(f"    Could not read image. Skipping.")
            return False
        
        img_h_full, img_w_full = image_bgr_full.shape[:2]
        gray_image_full = cv2.cvtColor(image_bgr_full, cv2.COLOR_BGR2GRAY)

        faces_full = face_detector_dlib(gray_image_full)
        if not faces_full:
            print("    No faces detected in original image.")
            return "NO_FACE_ORIGINAL" # Special string to count this case
        
        face = faces_full[0] # Process first detected face
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        face_w, face_h = r - l, b - t
        pad_w, pad_h = int(face_w * FACE_CROP_PADDING_FACTOR), int(face_h * FACE_CROP_PADDING_FACTOR)
        crop_l, crop_t = max(0, l - pad_w), max(0, t - pad_h)
        crop_r, crop_b = min(img_w_full, r + pad_w), min(img_h_full, b + pad_h)
        
        face_crop_orig_scale_bgr = image_bgr_full[crop_t:crop_b, crop_l:crop_r]
        face_crop_orig_scale_gray = gray_image_full[crop_t:crop_b, crop_l:crop_r]
        
        if face_crop_orig_scale_bgr.size == 0:
            print("    Padded face crop resulted in an empty image.")
            return False

        final_resized_crop_bgr = cv2.resize(face_crop_orig_scale_bgr, (TARGET_SIZE_W, TARGET_SIZE_H), interpolation=cv2.INTER_LINEAR)
        final_resized_crop_gray = cv2.resize(face_crop_orig_scale_gray, (TARGET_SIZE_W, TARGET_SIZE_H), interpolation=cv2.INTER_LINEAR)

        output_canvas = np.full((TARGET_SIZE_H, TARGET_SIZE_W, 3), 255, dtype=np.uint8)
        
        faces_in_resized_crop = face_detector_dlib(final_resized_crop_gray)
        face_mask_for_hough = np.zeros(final_resized_crop_gray.shape[:2], dtype=np.uint8)

        if not faces_in_resized_crop:
            print("    No face in resized crop for Dlib landmarks. Mask will be all white.")
            face_mask_for_hough.fill(255) # Process whole resized image if landmarks fail here
            landmarks_detected_in_crop = False
        else:
            landmarks_detected_in_crop = True
            landmarks_face = sorted(faces_in_resized_crop, key=lambda rect: rect.width() * rect.height(), reverse=True)[0]
            landmarks = landmark_predictor(final_resized_crop_gray, landmarks_face)
            shape_np = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], dtype="int")

            for _, indices in DLIB_LINE_SETS.items():
                points = np.array([shape_np[i] for i in indices], dtype=np.int32)
                cv2.polylines(output_canvas, [points], isClosed=False, color=LANDMARK_LINE_COLOR, thickness=LANDMARK_LINE_THICKNESS)
            
            hull = cv2.convexHull(shape_np)
            cv2.fillConvexPoly(face_mask_for_hough, hull, 255)
        
        if APPLY_HSV_VALUE_NORMALIZATION:
            hsv_image = cv2.cvtColor(final_resized_crop_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            v[:] = HSV_CONSTANT_V_VALUE
            modified_hsv_image = cv2.merge([h, s, v])
            modified_bgr_image = cv2.cvtColor(modified_hsv_image, cv2.COLOR_HSV2BGR)
            processed_gray_for_hough = cv2.cvtColor(modified_bgr_image, cv2.COLOR_BGR2GRAY)
            if SAVE_DEBUG_IMAGES:
                cv2.imwrite(os.path.join(DEBUG_DIRS["hsv_mod_gray"], f"hsv_gray_{base_filename}.png"), processed_gray_for_hough)
        else:
            processed_gray_for_hough = final_resized_crop_gray.copy()
            if APPLY_CLAHE_HOUGH:
                clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT_HOUGH, tileGridSize=CLAHE_TILE_GRID_SIZE_HOUGH)
                processed_gray_for_hough = clahe.apply(processed_gray_for_hough)
            if APPLY_GAUSSIAN_BLUR_HOUGH and BLUR_KERNEL_HOUGH is not None:
                processed_gray_for_hough = cv2.GaussianBlur(processed_gray_for_hough, BLUR_KERNEL_HOUGH, 0)
            if APPLY_BILATERAL_FILTER:
                bilateral_filtered_image = cv2.bilateralFilter(processed_gray_for_hough, 
                                                               BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
                processed_gray_for_hough = bilateral_filtered_image
                if SAVE_DEBUG_IMAGES:
                    cv2.imwrite(os.path.join(DEBUG_DIRS["bilateral"], f"bilateral_{base_filename}.png"), processed_gray_for_hough)

        image_for_canny_final = np.full_like(processed_gray_for_hough, 255, dtype=np.uint8)
        image_for_canny_final = np.where(face_mask_for_hough == 255, processed_gray_for_hough, 255) 
        
        edges_for_hough = cv2.Canny(image_for_canny_final, CANNY_THRESHOLD_1_HOUGH, CANNY_THRESHOLD_2_HOUGH)
        
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_DIRS["canny"], f"canny_{base_filename}.png"), edges_for_hough)
        
        lines = cv2.HoughLinesP(edges_for_hough, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
        
        hough_lines_drawn = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output_canvas, (x1, y1), (x2, y2), HOUGH_LINE_COLOR, HOUGH_LINE_THICKNESS)
                hough_lines_drawn += 1
        
        cv2.imwrite(output_image_path, output_canvas)
        print(f"    Saved trace: {os.path.basename(output_image_path)} ({hough_lines_drawn} Hough lines)")
        return "SUCCESS" if landmarks_detected_in_crop else "NO_FACE_RESIZED"

    except Exception as e:
        print(f"    ERROR processing {os.path.basename(input_image_path)}: {e}")
        traceback.print_exc()
        return False

# --- Main Script Logic ---
def run_batch_processing():
    global DEBUG_DIRS # Allow modification of the global dict

    if face_detector_dlib is None or landmark_predictor is None:
        print("Critical Dlib components not initialized. Exiting.")
        return

    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    DEBUG_DIRS = { # Define paths for debug images
        "canny": os.path.join(OUTPUT_FOLDER, "debug_canny"),
        "hsv_mod_gray": os.path.join(OUTPUT_FOLDER, "debug_hsv_mod_gray"),
        "bilateral": os.path.join(OUTPUT_FOLDER, "debug_bilateral")
    }
    if SAVE_DEBUG_IMAGES:
        for _, path in DEBUG_DIRS.items():
            if not os.path.exists(path): os.makedirs(path)

    print(f"Starting batch processing...")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}'")
    print(f"HSV Value Normalization: {APPLY_HSV_VALUE_NORMALIZATION}")
    if APPLY_HSV_VALUE_NORMALIZATION:
        print(f"  Constant V Channel Value: {HSV_CONSTANT_V_VALUE}")
    else:
        print(f"  CLAHE: {APPLY_CLAHE_HOUGH}, Gaussian Blur: {APPLY_GAUSSIAN_BLUR_HOUGH}, Bilateral Filter: {APPLY_BILATERAL_FILTER}")
    print(f"Face Crop Padding: {FACE_CROP_PADDING_FACTOR*100}%")
    print(f"Target Size: {TARGET_SIZE_W}x{TARGET_SIZE_H}")
    print(f"Canny Thresh: L={CANNY_THRESHOLD_1_HOUGH} H={CANNY_THRESHOLD_2_HOUGH}")
    print(f"Hough Thresh: {HOUGH_THRESHOLD}, MinLen: {HOUGH_MIN_LINE_LENGTH}, MaxGap: {HOUGH_MAX_LINE_GAP}")


    processed_count = 0
    general_skip_count = 0 # For read errors or empty crops
    no_face_original_count = 0
    no_face_resized_count = 0


    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename_base = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}.png")
        
        print(f"\nProcessing file: {filename}")
        result = generate_single_image_trace(input_path, output_path, base_filename=output_filename_base)

        if result == "SUCCESS":
            processed_count += 1
        elif result == "NO_FACE_ORIGINAL":
            no_face_original_count += 1
        elif result == "NO_FACE_RESIZED":
            # This image was processed up to a point, but Dlib landmarks weren't drawn for the mask.
            # It still produces an output (potentially with Hough lines on unmasked face or full crop).
            # Decide if this should be counted as "processed" or a specific type of skip.
            # For now, let's say it's processed but with a warning.
            processed_count += 1 # It still generates an output
            no_face_resized_count +=1
        else: # General skip / False
            general_skip_count += 1
            
    print(f"\n--- Batch Processing Summary ---")
    print(f"Successfully processed (with/without full Dlib landmarks on crop): {processed_count}")
    print(f"  (Specifically, Dlib landmarks on crop failed for: {no_face_resized_count} images)")
    print(f"Skipped (no face in original): {no_face_original_count}")
    print(f"Skipped (other errors like read/empty crop): {general_skip_count}")
    if SAVE_DEBUG_IMAGES:
        print(f"Debug images saved in subfolders of: {OUTPUT_FOLDER}")
    print("Batch processing complete.")

if __name__ == "__main__":
    run_batch_processing()