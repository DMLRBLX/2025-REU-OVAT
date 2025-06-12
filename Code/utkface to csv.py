import csv
import os

# Data to be written
data = [
    ["Image", "Age"]
]

# File path for the CSV file
csv_file_path = "UTKFace Features.csv"

INPUT_FOLDER = "Code/INPUT [UTKFace]"

for root_dir, _, files in os.walk(INPUT_FOLDER):
    for image in files:
        age = int(image.split('_')[0])
        data.append(["Code/INPUT [UTKFace]/" + image, age])
        print(f"[{image}, {age}]")
                    
# Open the file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    
    # Write data to the CSV file
    writer.writerows(data)

print(f"UTKFace features created successfully!")