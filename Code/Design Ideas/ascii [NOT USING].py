import os
import glob
from PIL import Image, UnidentifiedImageError

# --- ASCII Art Configuration ---

# Characters sorted from darkest/densest to lightest/sparsest.
# You can experiment with different ramps.
# ASCII_CHARS_RAMP = "@%#*+=-:. "  # Simple ramp
ASCII_CHARS_RAMP = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. " # More detailed

# Character aspect ratio (width / height). Most characters are taller than wide.
# A value of 0.5 means width is half the height. Adjust for your font.
CHARACTER_ASPECT_RATIO = 0.55

def image_to_ascii(image_path, output_width_chars=100, char_ramp=ASCII_CHARS_RAMP):
    """
    Converts an image to ASCII art.

    Args:
        image_path (str): Path to the input image.
        output_width_chars (int): Desired width of the ASCII art in characters.
        char_ramp (str): String of characters to use for mapping brightness,
                         from darkest to lightest.

    Returns:
        str: The generated ASCII art string, or None if an error occurs.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except UnidentifiedImageError:
        print(f"Error: Could not read or identify image at {image_path}.")
        return None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    # Convert to grayscale
    img = img.convert("L")  # "L" mode is grayscale

    # Calculate new dimensions based on output width and character aspect ratio
    original_width, original_height = img.size
    aspect_ratio = original_height / float(original_width)
    
    # New height considers the character aspect ratio to make the ASCII art look proportional
    new_height_chars = int(aspect_ratio * output_width_chars * CHARACTER_ASPECT_RATIO)
    if new_height_chars <= 0: new_height_chars = 1 # Ensure at least 1 character height

    # Resize the image to the target character dimensions
    img_resized = img.resize((output_width_chars, new_height_chars))

    # Get pixel data
    pixels = img_resized.load()

    # Build ASCII string
    ascii_str = ""
    num_ramp_chars = len(char_ramp)

    for y in range(new_height_chars):
        for x in range(output_width_chars):
            brightness = pixels[x, y]  # Grayscale value (0-255)
            # Map brightness to an ASCII character
            # The darkest char in ramp is for lowest brightness, lightest for highest
            char_index = int((brightness / 255) * (num_ramp_chars - 1))
            ascii_str += char_ramp[num_ramp_chars - 1 - char_index] # Darkest char for low brightness
        ascii_str += "\n"

    return ascii_str

def batch_process_to_ascii(input_folder, output_folder, output_width_chars=100):
    """
    Converts all PNG images in input_folder to ASCII art .txt files in output_folder.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    search_pattern_lower = os.path.join(input_folder, '*.png')
    search_pattern_upper = os.path.join(input_folder, '*.PNG')
    jpg_pattern_lower = os.path.join(input_folder, '*.jpg')
    jpg_pattern_upper = os.path.join(input_folder, '*.JPG')
    jpeg_pattern_lower = os.path.join(input_folder, '*.jpeg')
    jpeg_pattern_upper = os.path.join(input_folder, '*.JPEG')


    image_files = list(set(
        glob.glob(search_pattern_lower) + glob.glob(search_pattern_upper) +
        glob.glob(jpg_pattern_lower) + glob.glob(jpg_pattern_upper) +
        glob.glob(jpeg_pattern_lower) + glob.glob(jpeg_pattern_upper)
    ))

    if not image_files:
        print(f"No PNG, JPG, or JPEG files found in '{input_folder}'.")
        return

    print(f"Found {len(image_files)} image files to process.")
    processed_count = 0
    failed_count = 0

    for i, input_image_path in enumerate(image_files):
        base_filename = os.path.basename(input_image_path)
        name_part, _ = os.path.splitext(base_filename)
        output_txt_filename = name_part + ".txt"
        output_txt_path = os.path.join(output_folder, output_txt_filename)

        print(f"Processing image {i+1}/{len(image_files)}: {base_filename} ... ", end="", flush=True)
        
        ascii_art = image_to_ascii(input_image_path, output_width_chars)

        if ascii_art:
            try:
                with open(output_txt_path, "w") as f:
                    f.write(ascii_art)
                print("Done.")
                processed_count += 1
            except Exception as e:
                print(f"Failed to write {output_txt_path}: {e}")
                failed_count += 1
        else:
            print("Failed (conversion error).")
            failed_count += 1

    print(f"\nBatch ASCII conversion complete.")
    print(f"Successfully converted: {processed_count} images.")
    print(f"Failed to convert: {failed_count} images.")


if __name__ == '__main__':
    # --- Option 1: Process a Single Image ---
    # single_image_path = "input_image.png" # <--- CHANGE THIS
    # ascii_width = 120
    #
    # # Create a dummy input image if the specified one doesn't exist for quick testing
    # if not os.path.exists(single_image_path):
    #     print(f"Single image '{single_image_path}' not found. Creating a dummy image.")
    #     try:
    #         from PIL import ImageDraw
    #         temp_img = Image.new("RGB", (200, 150), "lightblue")
    #         draw = ImageDraw.Draw(temp_img)
    #         draw.ellipse((30, 30, 170, 120), fill="salmon", outline="red")
    #         draw.text((50,60), "Hello", fill="black")
    #         temp_img.save(single_image_path)
    #     except Exception as e:
    #         print(f"Could not create dummy single image: {e}")

    # print(f"\n--- Processing Single Image: {single_image_path} ---")
    # result_ascii = image_to_ascii(single_image_path, output_width_chars=ascii_width)
    # if result_ascii:
    #     print("\nASCII Art Output:\n")
    #     print(result_ascii)
    #     # Optionally save to a file
    #     # with open("output_ascii_art.txt", "w") as f:
    #     #     f.write(result_ascii)
    #     # print("\n(Saved to output_ascii_art.txt)")


    # --- Option 2: Batch Process a Folder ---
    # Comment out Option 1 if using Option 2, and vice-versa.

    # --- IMPORTANT: SET THESE FOLDER PATHS FOR BATCH MODE ---
    input_images_folder = "Design Ideas/INPUT"   # <--- CHANGE THIS
    output_ascii_folder = "Design Ideas/OUTPUT" # <--- CHANGE THIS
    ascii_art_output_width = 100 # Number of characters for the width

    # Create dummy input folder and image for quick testing if they don't exist
    if not os.path.exists(input_images_folder):
        print(f"Input folder '{input_images_folder}' not found. Creating it with a dummy image.")
        os.makedirs(input_images_folder)
        try:
            from PIL import ImageDraw
            temp_img_batch = Image.new("RGB", (300, 200), "lightgreen")
            draw_batch = ImageDraw.Draw(temp_img_batch)
            gradient_steps = 100
            for i in range(gradient_steps):
                # Simple horizontal gradient
                color_val = int(255 * (i / float(gradient_steps)))
                draw_batch.line([(i * (300/gradient_steps), 0), (i * (300/gradient_steps), 200)], fill=(color_val, color_val, 255-color_val))
            draw_batch.ellipse((50, 50, 250, 150), outline="black", width=5)
            temp_img_batch.save(os.path.join(input_images_folder, "dummy_batch_image.png"))
            print(f"Dummy image 'dummy_batch_image.png' created in '{input_images_folder}'.")
        except Exception as e_dummy_batch:
            print(f"Could not create dummy batch image: {e_dummy_batch}.")

    print(f"\n--- Starting Batch ASCII Art Conversion ---")
    print(f"Input folder: '{os.path.abspath(input_images_folder)}'")
    print(f"Output folder: '{os.path.abspath(output_ascii_folder)}'")
    
    batch_process_to_ascii(
        input_images_folder,
        output_ascii_folder,
        output_width_chars=ascii_art_output_width
    )
