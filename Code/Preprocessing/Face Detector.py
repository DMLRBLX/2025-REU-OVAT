import cv2
import dlib
import numpy as np
import os
import traceback

# --- Configuration ---
INPUT_PARENT_FOLDER = "Code/UTKFace"
FLAT_OUTPUT_FOLDER = "Code/INPUT [UTKFace]"
# Ensure this output folder is at a location where you want it.
# If it's relative, it will be created relative to where the script is run.
# For example, if you want it next to PARENT_FOLDER, make sure your paths reflect that
# or use absolute paths.

# Target size for the resized face crop
TARGET_SIZE_W = 220
TARGET_SIZE_H = 220

# Padding around the detected face bounding box
FACE_CROP_PADDING_FACTOR = 0.30

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

# Output image format and naming
OUTPUT_IMAGE_EXTENSION = ".png" # Standardize output to PNG
OUTPUT_FILENAME_PADDING = 5     # e.g., 00001.png, 00010.png

# --- Dlib Initialization ---
try:
    face_detector_dlib = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Error initializing Dlib face detector: {e}")
    print("Please ensure Dlib is correctly installed.")
    face_detector_dlib = None

# --- Single Image Processing Function (largely unchanged) ---
def crop_resize_single_face(input_image_path, output_image_path):
    """
    Detects a face, crops, resizes, and saves.
    Returns a status: "SUCCESS", "NO_FACE", "ERROR_READ", "ERROR_PROCESS".
    """
    if face_detector_dlib is None:
        return "ERROR_PROCESS" # Dlib not available

    try:
        image_bgr_full = cv2.imread(input_image_path)
        if image_bgr_full is None:
            # print(f"    Could not read image: {os.path.basename(input_image_path)}") # Verbose
            return "ERROR_READ"

        img_h_full, img_w_full = image_bgr_full.shape[:2]
        gray_image_full = cv2.cvtColor(image_bgr_full, cv2.COLOR_BGR2GRAY)
        faces_full = face_detector_dlib(gray_image_full, 1)

        if not faces_full:
            # print(f"    No faces detected in: {os.path.basename(input_image_path)}") # Verbose
            return "NO_FACE"
        
        face = faces_full[0]
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        face_w, face_h = r - l, b - t
        pad_w, pad_h = int(face_w * FACE_CROP_PADDING_FACTOR), int(face_h * FACE_CROP_PADDING_FACTOR)
        crop_l, crop_t = max(0, l - pad_w), max(0, t - pad_h)
        crop_r, crop_b = min(img_w_full, r + pad_w), min(img_h_full, b + pad_h)
        face_crop_bgr = image_bgr_full[crop_t:crop_b, crop_l:crop_r]

        if face_crop_bgr.size == 0:
            return "ERROR_PROCESS"

        resized_face_bgr = cv2.resize(face_crop_bgr, (TARGET_SIZE_W, TARGET_SIZE_H), interpolation=cv2.INTER_LINEAR)
        
        # Ensure the output directory for this specific file exists (though for flat structure, it's just one dir)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, resized_face_bgr)
        return "SUCCESS"

    except Exception as e:
        print(f"    ERROR processing {os.path.basename(input_image_path)}: {e}")
        # traceback.print_exc()
        return "ERROR_PROCESS"

# --- Main Batch Processing Function with Flattened Output ---
def batch_process_to_flat_folder(input_parent_dir, flat_output_dir_name):
    if face_detector_dlib is None:
        print("Dlib face detector not initialized. Cannot start batch processing.")
        return

    if not os.path.isdir(input_parent_dir):
        print(f"Error: Input parent folder '{input_parent_dir}' not found.")
        return

    # Create the single flat output directory
    if not os.path.exists(flat_output_dir_name):
        os.makedirs(flat_output_dir_name)
        print(f"Created output folder: '{os.path.abspath(flat_output_dir_name)}'")
    else:
        print(f"Outputting to existing folder: '{os.path.abspath(flat_output_dir_name)}'")


    print(f"\nStarting batch processing...")
    print(f"Input Parent Folder: '{os.path.abspath(input_parent_dir)}'")
    print(f"Target Cropped Face Size: {TARGET_SIZE_W}x{TARGET_SIZE_H}")
    print(f"Face Crop Padding Factor: {FACE_CROP_PADDING_FACTOR}")

    stats = {"SUCCESS": 0, "NO_FACE": 0, "ERROR_READ": 0, "ERROR_PROCESS": 0, "TOTAL_FILES_SCANNED":0}
    image_output_counter = 1

    for root_dir, _, files in os.walk(input_parent_dir):
        # Sort files to ensure a somewhat consistent processing order across runs,
        # though os.walk order isn't guaranteed across systems for directories.
        # For files within a directory, sorting helps.
        files.sort()
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                stats["TOTAL_FILES_SCANNED"] += 1
                input_image_path = os.path.join(root_dir, filename)
                
                # Generate new filename based on counter
                output_filename = f"{image_output_counter:0{OUTPUT_FILENAME_PADDING}d}{OUTPUT_IMAGE_EXTENSION}"
                output_image_path = os.path.join(flat_output_dir_name, filename)

                print(f"Processing: {input_image_path} -> {filename}")
                status = crop_resize_single_face(input_image_path, output_image_path)
                stats[status] += 1

                if status == "SUCCESS":
                    image_output_counter += 1 # Only increment counter if image was successfully processed and saved
            
    print(f"\n--- Batch Processing Summary ---")
    for status_key, count in stats.items():
        print(f"{status_key.replace('_', ' ').title()}: {count}")
    print("Batch processing complete.")


if __name__ == "__main__":
    # The FLAT_OUTPUT_FOLDER will be created if it doesn't exist.

    # Create dummy input structure for quick testing
    if not os.path.exists(INPUT_PARENT_FOLDER):
        print(f"Input folder '{INPUT_PARENT_FOLDER}' not found. Creating it with dummy structure and images.")
        os.makedirs(INPUT_PARENT_FOLDER)
        child_folders = ["CHILD_FOLDER_1", "CHILD_FOLDER_2", "CHILD_FOLDER_1/SUB_CHILD_A"]
        for cf in child_folders:
            os.makedirs(os.path.join(INPUT_PARENT_FOLDER, cf), exist_ok=True)
        dummy_img_data = np.zeros((400, 600, 3), dtype=np.uint8)
        for y_coord in range(400):
            for x_coord in range(600):
                dummy_img_data[y_coord, x_coord] = (x_coord % 256, y_coord % 256, (x_coord+y_coord) % 256)
        cv2.imwrite(os.path.join(INPUT_PARENT_FOLDER, "CHILD_FOLDER_1", "dummy1.jpg"), dummy_img_data)
        cv2.imwrite(os.path.join(INPUT_PARENT_FOLDER, "CHILD_FOLDER_1", "dummy2.png"), dummy_img_data[::2,::2])
        cv2.imwrite(os.path.join(INPUT_PARENT_FOLDER, "CHILD_FOLDER_2", "another_dummy.jpeg"), dummy_img_data)
        cv2.imwrite(os.path.join(INPUT_PARENT_FOLDER, "CHILD_FOLDER_1/SUB_CHILD_A", "sub_dummy.jpg"), dummy_img_data)
        print("Dummy structure created. These dummy images likely DON'T contain faces.")

    if face_detector_dlib:
        batch_process_to_flat_folder(INPUT_PARENT_FOLDER, FLAT_OUTPUT_FOLDER)
    else:
        print("Cannot run batch processing as Dlib face detector failed to initialize.")