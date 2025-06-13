import cv2
import numpy as np
import os
import traceback

# --- Configuration ---
INPUT_FOLDER = "Code/Input/input_fixed/INPUT [UTKFace]" # Source images
OUTPUT_FOLDER = "Code/preprocessing_output/utkface_output/OUTPUT [GREYSCALE]" # Output for this version

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

# --- Single Image Grayscale Conversion Function ---
def convert_to_grayscale_single_image(input_image_path, output_image_path):
    """
    Converts a single image to grayscale and saves it.
    Returns True on success, False on failure.
    """
    try:
        # Read the image in color (default)
        image_bgr = cv2.imread(input_image_path)
        if image_bgr is None:
            print(f"    Failed to read image: {os.path.basename(input_image_path)}")
            return False

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Ensure the output directory for this specific file exists
        output_dir_for_file = os.path.dirname(output_image_path)
        if not os.path.exists(output_dir_for_file):
            os.makedirs(output_dir_for_file)

        # Save the grayscale image
        cv2.imwrite(output_image_path, grayscale_image)
        # print(f"    Saved grayscale image: {os.path.basename(output_image_path)}") # Can be verbose
        return True

    except Exception as e:
        print(f"    ERROR processing {os.path.basename(input_image_path)}: {e}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        return False

# --- Main Batch Processing Function ---
def batch_convert_to_grayscale(input_parent_dir, output_parent_dir):
    if not os.path.isdir(input_parent_dir):
        print(f"Error: Input parent folder '{input_parent_dir}' not found.")
        return

    print(f"Starting grayscale conversion...")
    print(f"Input Parent Folder: '{os.path.abspath(input_parent_dir)}'")
    print(f"Output Parent Folder: '{os.path.abspath(output_parent_dir)}'")

    stats = {"SUCCESS": 0, "FAILED": 0, "TOTAL_FILES_SCANNED": 0}

    for root_dir, _, files in os.walk(input_parent_dir):
        # Determine the corresponding output subdirectory
        # This replicates the input folder structure in the output
        relative_path = os.path.relpath(root_dir, input_parent_dir)
        output_subdir_path = os.path.join(output_parent_dir, relative_path)

        files.sort() # Process files in a consistent order within each directory
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                stats["TOTAL_FILES_SCANNED"] += 1
                input_image_path = os.path.join(root_dir, filename)
                
                # Output filename will be the same as input, in the corresponding output subfolder
                output_image_path = os.path.join(output_subdir_path, filename)

                print(f"Processing: {input_image_path} -> Greyscale")
                if convert_to_grayscale_single_image(input_image_path, output_image_path):
                    stats["SUCCESS"] += 1
                else:
                    stats["FAILED"] += 1
            
    print(f"\n--- Greyscale Conversion Summary ---")
    print(f"Total Image Files Scanned: {stats['TOTAL_FILES_SCANNED']}")
    print(f"Successfully Converted: {stats['SUCCESS']}")
    print(f"Failed to Convert: {stats['FAILED']}")
    print("Batch greyscale conversion complete.")


if __name__ == "__main__":
    # --- Set your input and output parent folder paths in the Configuration section ---

    # Create dummy input structure for quick testing if INPUT_FOLDER doesn't exist
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found. Creating it with dummy structure and images.")
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        
        # Create some child folders
        child_folders_for_dummy = ["Subfolder1", "Subfolder2", "Subfolder1/Nested"]
        for cf in child_folders_for_dummy:
            os.makedirs(os.path.join(INPUT_FOLDER, cf), exist_ok=True)

        # Create a simple dummy color image (gradient) using OpenCV
        dummy_color_img_data = np.zeros((100, 150, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(150):
                dummy_color_img_data[y, x] = (x * 255 // 150, y * 255 // 100, (x + y) * 255 // 250) # BGR
        
        # Save dummy images in the child folders
        cv2.imwrite(os.path.join(INPUT_FOLDER, "Subfolder1", "color_dummy1.jpg"), dummy_color_img_data)
        cv2.imwrite(os.path.join(INPUT_FOLDER, "Subfolder1", "color_dummy2.png"), dummy_color_img_data[::2,::2])
        cv2.imwrite(os.path.join(INPUT_FOLDER, "Subfolder2", "another_color_dummy.jpeg"), dummy_color_img_data)
        cv2.imwrite(os.path.join(INPUT_FOLDER, "Subfolder1/Nested", "nested_color_dummy.bmp"), dummy_color_img_data)
        print(f"Dummy structure and color images created in '{INPUT_FOLDER}'.")

    batch_convert_to_grayscale(INPUT_FOLDER, OUTPUT_FOLDER)