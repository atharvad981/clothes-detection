from PIL import Image, UnidentifiedImageError
import os

# Define the input and output folders
input_folders = ["dataset/Men", "dataset/Women", "dataset/Child"]
output_folder = "resized_dataset"

# Create the resized_dataset folder structure
os.makedirs(output_folder, exist_ok=True)
for folder in input_folders:
    class_name = os.path.basename(folder)
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# Resize images
for folder in input_folders:
    class_name = os.path.basename(folder)
    output_class_folder = os.path.join(output_folder, class_name)
    for filename in os.listdir(folder):
        input_path = os.path.join(folder, filename)
        output_path = os.path.join(output_class_folder, filename)
        try:
            if filename.endswith((".jpg", ".jpeg", ".png")):
                with Image.open(input_path) as img:
                    img = img.resize((224, 224))  # Resize to 224x224
                    img.save(output_path)
        except UnidentifiedImageError:
            print(f"Skipping invalid image: {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

print("All images resized successfully!")
