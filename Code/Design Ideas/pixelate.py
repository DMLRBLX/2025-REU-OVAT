import cv2
import dlib
import numpy as np
import os
import math
import traceback

# --- Configuration ---
INPUT_FOLDER = "Code/INPUT [FIXED]" # Source images
OUTPUT_FOLDER = "Design Ideas/OUTPUT [PIXELATED]" # Output folder for pixel art
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" # Dlib face detector (bounding box needed)

# Face detection and initial resize settings (consistent with previous scripts)
TARGET_SIZE_W = 220 # Width of the face image before pixelization
TARGET_SIZE_H = 220 # Height of the face image before pixelization
FACE_CROP_PADDING_FACTOR = 0.30 # Padding around detected face

# --- Pixel Art Specific Parameters ---
PIXEL_ART_BLOCK_SIZE = 48 # e.g., 32, 48, 64. Face will be downscaled to this WxH.
                          # Smaller means more blocky.
NUM_COLORS_KMEANS = 16    # Number of colors in the final pixel art palette (e.g., 8, 16, 32)

# Output size for the final pixel art (can be same as TARGET_SIZE or larger for viewing)
FINAL_OUTPUT_SIZE_W = 220
FINAL_OUTPUT_SIZE_H = 220


SAVE_DEBUG_IMAGES = True # Set to False to disable saving intermediate steps
DEBUG_DIRS = {}
# --- Dlib Initialization ---
if not os.path.exists(PREDICTOR_PATH):
    print(f"WARNING: Dlib landmark predictor not found at '{PREDICTOR_PATH}'. Face detection might be less robust or only bounding box used if this script were to use landmarks.")
    # For this script, only the face detector is strictly needed for the bounding box.
    # If shape_predictor_68_face_landmarks.dat is used by get_frontal_face_detector implicitly, it's an issue.
    # However, get_frontal_face_detector() itself does not require the landmark model.
    # So, we only strictly need dlib library installed.
    # For robustness, let's assume the file might be used by some dlib versions/setups,
    # but the core logic here only needs the detector.
    # If the user has been running previous scripts, they likely have it.
    pass # landmark_predictor not used in this script, only face_detector_dlib

try:
    face_detector_dlib = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Error initializing Dlib face detector: {e}")
    print("Please ensure Dlib is correctly installed.")
    face_detector_dlib = None # Will cause script to exit gracefully later


# --- Single Image Pixel Art Function ---
def generate_single_image_pixel_art(input_image_path, output_image_path, base_filename="image"):
    """
    Detects a face, crops it, resizes it, then converts it to pixel art.
    Returns a status string or False on failure.
    """
    if face_detector_dlib is None:
        print("  Dlib face detector not initialized. Cannot process.")
        return False

    try:
        print(f"  Starting pixel art processing for: {os.path.basename(input_image_path)}")
        image_bgr_full = cv2.imread(input_image_path)
        if image_bgr_full is None:
            print(f"    Could not read image. Skipping.")
            return False
        
        img_h_full, img_w_full = image_bgr_full.shape[:2]
        gray_image_full = cv2.cvtColor(image_bgr_full, cv2.COLOR_BGR2GRAY)

        # 1. Detect face in the original full image
        faces_full = face_detector_dlib(gray_image_full)
        if not faces_full:
            print("    No faces detected in original image.")
            return "NO_FACE_ORIGINAL"
        
        face = faces_full[0] # Process first detected face
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        face_w, face_h = r - l, b - t
        pad_w = int(face_w * FACE_CROP_PADDING_FACTOR)
        pad_h = int(face_h * FACE_CROP_PADDING_FACTOR)
        crop_l, crop_t = max(0, l - pad_w), max(0, t - pad_h)
        crop_r, crop_b = min(img_w_full, r + pad_w), min(img_h_full, b + pad_h)
        
        face_crop_bgr = image_bgr_full[crop_t:crop_b, crop_l:crop_r]
        if face_crop_bgr.size == 0:
            print("    Padded face crop resulted in an empty image.")
            return False

        # 2. Resize this padded face crop to our intermediate TARGET_SIZE
        resized_face_bgr = cv2.resize(face_crop_bgr, (TARGET_SIZE_W, TARGET_SIZE_H), interpolation=cv2.INTER_LINEAR)
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_DIRS["resized_face"], f"resized_face_{base_filename}.png"), resized_face_bgr)

        # --- Pixel Art Conversion Steps ---
        # a. Downscale to Pixel Art Resolution
        small_pixel_img = cv2.resize(resized_face_bgr, (PIXEL_ART_BLOCK_SIZE, PIXEL_ART_BLOCK_SIZE), interpolation=cv2.INTER_LINEAR) # INTER_AREA is also good for downsampling
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_DIRS["downscaled"], f"downscaled_{PIXEL_ART_BLOCK_SIZE}x{PIXEL_ART_BLOCK_SIZE}_{base_filename}.png"), small_pixel_img)

        # b. Color Quantization using K-Means
        # Reshape image into a list of BGR pixels
        pixels = small_pixel_img.reshape((-1, 3)) # Shape: (N, 3) where N = width*height
        pixels = np.float32(pixels) # Convert to float32 for K-Means

        # Define criteria and K for K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5) # (type, max_iter, epsilon)
                                                                                # Might need to adjust max_iter (e.g. 10-100) & epsilon (e.g. 0.1-1.0)
        
        compactness, labels, centers = cv2.kmeans(pixels, NUM_COLORS_KMEANS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                                                # attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS (or cv2.KMEANS_PP_CENTERS)
        
        centers = np.uint8(centers) # Convert K-Means centers (palette colors) to uint8
        quantized_pixels = centers[labels.flatten()] # Map each original pixel to its new palette color
        quantized_small_pixel_img = quantized_pixels.reshape(small_pixel_img.shape) # Reshape back to small image dimensions

        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_DIRS["quantized"], f"quantized_{NUM_COLORS_KMEANS}colors_{base_filename}.png"), quantized_small_pixel_img)

        # c. Upscale with Nearest Neighbor interpolation to keep the blocky look
        final_pixel_art_output = cv2.resize(quantized_small_pixel_img, 
                                            (FINAL_OUTPUT_SIZE_W, FINAL_OUTPUT_SIZE_H), 
                                            interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(output_image_path, final_pixel_art_output)
        print(f"    Saved pixel art: {os.path.basename(output_image_path)}")
        return "SUCCESS"

    except Exception as e:
        print(f"    ERROR processing {os.path.basename(input_image_path)}: {e}")
        traceback.print_exc()
        return False

# --- Main Script Logic ---
def run_batch_pixel_art_conversion():
    global DEBUG_DIRS

    if face_detector_dlib is None:
        print("Critical Dlib face detector not initialized. Exiting.")
        return

    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    DEBUG_DIRS = {
        "resized_face": os.path.join(OUTPUT_FOLDER, "debug_01_resized_face"),
        "downscaled": os.path.join(OUTPUT_FOLDER, "debug_02_downscaled"),
        "quantized": os.path.join(OUTPUT_FOLDER, "debug_03_quantized_colors")
    }
    if SAVE_DEBUG_IMAGES:
        for _, path in DEBUG_DIRS.items():
            if not os.path.exists(path): os.makedirs(path)

    print(f"Starting batch pixel art conversion...")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}'")
    print(f"Face Crop Padding: {FACE_CROP_PADDING_FACTOR*100}%")
    print(f"Intermediate Face Resize: {TARGET_SIZE_W}x{TARGET_SIZE_H}")
    print(f"Pixel Art Block Size (Downscale To): {PIXEL_ART_BLOCK_SIZE}x{PIXEL_ART_BLOCK_SIZE}")
    print(f"Number of Colors (K-Means): {NUM_COLORS_KMEANS}")
    print(f"Final Output Size: {FINAL_OUTPUT_SIZE_W}x{FINAL_OUTPUT_SIZE_H}")

    processed_count = 0
    general_skip_count = 0
    no_face_original_count = 0

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename_base = os.path.splitext(filename)[0]
        # Keep original extension for output if desired, or standardize to e.g. PNG
        output_ext = os.path.splitext(filename)[1] if os.path.splitext(filename)[1] else ".png"
        output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}{output_ext}")
        
        # print(f"\nProcessing file: {filename}") # Already printed in function
        result = generate_single_image_pixel_art(input_path, output_path, base_filename=output_filename_base)

        if result == "SUCCESS":
            processed_count += 1
        elif result == "NO_FACE_ORIGINAL":
            no_face_original_count += 1
        else: # General skip / False
            general_skip_count += 1
            
    print(f"\n--- Batch Processing Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no face in original): {no_face_original_count}")
    print(f"Skipped (other errors like read/empty crop): {general_skip_count}")
    if SAVE_DEBUG_IMAGES:
        print(f"Debug images saved in subfolders of: {OUTPUT_FOLDER}")
    print("Batch pixel art conversion complete.")

if __name__ == "__main__":
    # Create dummy input folder and image for quick testing
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created dummy input folder: '{INPUT_FOLDER}'.")
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw # Pillow for dummy
            temp_img = PILImage.new("RGB", (600, 800), (200, 220, 250))
            draw = PILImageDraw.Draw(temp_img)
            face_color = (255, 200, 180); eye_color = (50, 50, 50)
            draw.ellipse((150, 150, 450, 650), fill=face_color)
            draw.ellipse((220, 300, 280, 350), fill=eye_color)
            draw.ellipse((320, 300, 380, 350), fill=eye_color)
            draw.rectangle((270, 450, 330, 500), fill=eye_color)
            temp_img.save(os.path.join(INPUT_FOLDER, "dummy_pixel_face.jpg"))
            print(f"Dummy image 'dummy_pixel_face.jpg' created in '{INPUT_FOLDER}'.")
        except Exception as e_dummy:
            print(f"Could not create dummy image: {e_dummy}")
            
    run_batch_pixel_art_conversion()