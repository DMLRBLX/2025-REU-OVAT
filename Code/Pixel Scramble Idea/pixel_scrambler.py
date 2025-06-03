import cv2
import dlib
import numpy as np
import os
import random
import traceback

# --- Configuration ---
INPUT_FOLDER = "Design Ideas/INPUT"  # Folder with your input images
OUTPUT_FOLDER = "Design Ideas/OUTPUT [SCRAMBLED]" # Output folder for scrambled faces
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" # For Dlib face detection (detector only needed)

# Face detection and initial resize settings
TARGET_SIZE_W = 220 # Width of the face image before scrambling
TARGET_SIZE_H = 220 # Height of the face image before scrambling
FACE_CROP_PADDING_FACTOR = 0.30 # Padding around detected face

# Scrambling Configuration
SCRAMBLE_SEED = 42 # Change for different consistent scramble patterns

# --- Dlib Initialization ---
# Only the face detector is strictly needed for bounding box cropping
try:
    face_detector_dlib = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Error initializing Dlib face detector: {e}")
    print("Please ensure Dlib is correctly installed. This script requires it for face detection.")
    face_detector_dlib = None

# --- Single Image Processing Function ---
def scramble_single_face_cv2(input_image_path, output_image_path, base_filename="image"):
    """
    Detects a face, crops and resizes it, then scrambles its pixels consistently using OpenCV.
    Returns a status string ('SUCCESS', 'NO_FACE_ORIGINAL') or False on other errors.
    """
    if face_detector_dlib is None:
        print("  Dlib face detector not initialized. Cannot process.")
        return False

    try:
        print(f"  Starting scrambling for: {os.path.basename(input_image_path)}")
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
        
        # --- Pixel Scrambling Logic (using OpenCV/NumPy) ---
        height, width, channels = resized_face_bgr.shape
        total_pixels = width * height

        # Reshape the image data into a 2D array (total_pixels, num_channels)
        pixels_flat = resized_face_bgr.reshape((total_pixels, channels))

        # Generate a fixed permutation of pixel indices
        random.seed(SCRAMBLE_SEED)
        original_indices = list(range(total_pixels))
        shuffled_target_indices = original_indices[:] # Create a copy
        random.shuffle(shuffled_target_indices)

        # Create a new array for scrambled pixels
        scrambled_pixels_flat = np.zeros_like(pixels_flat)

        # Place original pixels into their new, scrambled positions
        # pixels_flat[i] goes to position shuffled_target_indices[i]
        for i in range(total_pixels):
            scrambled_pixels_flat[shuffled_target_indices[i]] = pixels_flat[i]
        
        # Reshape scrambled_pixels_flat back to image dimensions
        scrambled_img_bgr = scrambled_pixels_flat.reshape((height, width, channels))
        
        cv2.imwrite(output_image_path, scrambled_img_bgr)
        print(f"    Saved scrambled face: {os.path.basename(output_image_path)}")
        return "SUCCESS"

    except Exception as e:
        print(f"    ERROR processing {os.path.basename(input_image_path)}: {e}")
        traceback.print_exc()
        return False

# --- Batch Processing Function ---
def run_batch_scrambling_cv2():
    if face_detector_dlib is None:
        print("Critical Dlib face detector not initialized. Exiting batch process.")
        return

    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    print(f"Starting batch face scrambling...")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output folder: '{OUTPUT_FOLDER}'")
    print(f"Target Face Size for Scrambling: {TARGET_SIZE_W}x{TARGET_SIZE_H}")
    print(f"Face Crop Padding: {FACE_CROP_PADDING_FACTOR*100}%")
    print(f"Scramble Seed: {SCRAMBLE_SEED}")

    processed_count = 0
    general_skip_count = 0
    no_face_original_count = 0

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename_base = os.path.splitext(filename)[0]
        output_ext = os.path.splitext(filename)[1] if os.path.splitext(filename)[1] else ".png"
        output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}{output_ext}")
        
        # print(f"\nProcessing file: {filename}") # Reduced verbosity, printed in single function
        result = scramble_single_face_cv2(input_path, output_path, base_filename=output_filename_base)

        if result == "SUCCESS":
            processed_count += 1
        elif result == "NO_FACE_ORIGINAL":
            no_face_original_count += 1
        else: # General skip / False
            general_skip_count += 1
            
    print(f"\n--- Batch Scrambling Summary ---")
    print(f"Successfully processed and scrambled: {processed_count}")
    print(f"Skipped (no face in original image): {no_face_original_count}")
    print(f"Skipped (other errors like read/empty crop): {general_skip_count}")
    print("Batch scrambling complete.")

# --- Dummy Image Creation (Optional, uses OpenCV now) ---
def create_dummy_image_cv2(path, width=220, height=220):
    """Creates a simple gradient image for testing using OpenCV."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y_coord in range(height): # Renamed y to y_coord
        for x_coord in range(width): # Renamed x to x_coord
            r = int((x_coord / width) * 255)
            g = int((y_coord / height) * 255)
            b = 128
            img[y_coord, x_coord] = (b, g, r) # OpenCV is BGR
    cv2.imwrite(path, img)
    print(f"Dummy CV2 image created at {path}")


if __name__ == "__main__":
    # Create dummy input folder and image for quick testing
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created dummy input folder: '{INPUT_FOLDER}'.")
        # Create a dummy image using the new OpenCV dummy function
        # This dummy image will likely not have a "face" Dlib can detect well,
        # so it's mainly for testing the file handling part.
        # For actual face scrambling, use images with faces.
        dummy_image_path = os.path.join(INPUT_FOLDER, "dummy_scramble_test.png")
        create_dummy_image_cv2(dummy_image_path, width=600, height=400)
        # Add a more face-like dummy with Pillow if PIL is available for better testing of face detection
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw
            pil_dummy_path = os.path.join(INPUT_FOLDER, "dummy_scramble_face.jpg")
            temp_img = PILImage.new("RGB", (600, 800), (200, 220, 250))
            draw = PILImageDraw.Draw(temp_img)
            face_color = (255, 200, 180); eye_color = (50, 50, 50)
            draw.ellipse((150, 150, 450, 650), fill=face_color) # Face oval
            draw.ellipse((220, 300, 280, 350), fill=eye_color)  # Left eye
            draw.ellipse((320, 300, 380, 350), fill=eye_color)  # Right eye
            temp_img.save(pil_dummy_path)
            print(f"Dummy PIL face image created at {pil_dummy_path}")
        except ImportError:
            print("Pillow not found, skipping PIL-based dummy image creation.")
        except Exception as e_pil_dummy:
            print(f"Could not create PIL dummy image: {e_pil_dummy}")

    run_batch_scrambling_cv2()