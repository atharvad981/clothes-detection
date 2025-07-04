import os
import shutil
import pandas as pd  # Install pandas: pip install pandas

# Paths
dataset_folder = "path_to_dataset"  # Replace with your dataset folder path
output_folder = "dataset"  # This will be the final organized folder
metadata_file = os.path.join(dataset_folder, "labels.csv")  # Replace with your .csv file name

# Create class subfolders
classes = ["Men", "Women", "Child"]
for cls in classes:
    os.makedirs(os.path.join(output_folder, cls), exist_ok=True)

# Read metadata file
df = pd.read_csv(metadata_file)

# Move images into subfolders
for _, row in df.iterrows():
    src_path = os.path.join(dataset_folder, row["image_name"])
    dst_path = os.path.join(output_folder, row["class"], row["image_name"])
    shutil.move(src_path, dst_path)

print("Dataset organized successfully!")
