import torch
import cv2
import matplotlib.pyplot as plt

# Function to identify clothes in an image
def identify_clothes(image_path):
    # Load the pre-trained YOLOv5 model (COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Convert image to RGB (YOLOv5 uses RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference using YOLOv5
    results = model(image_rgb)

    # Extract detected objects
    detections = results.pandas().xyxy[0]  # Results as a pandas DataFrame
    clothes = detections[detections['name'].isin(['person', 'tie', 'suit', 'shirt', 'shoes', 'jacket'])]

    # Display results
    print("Identified Clothes:")
    for index, row in clothes.iterrows():
        print(f"- {row['name']} at coordinates {row['xmin']:.0f}, {row['ymin']:.0f}, {row['xmax']:.0f}, {row['ymax']:.0f}")

    # Visualize the image with bounding boxes
    results.show()

# Example usage
image_path = 'remond.jpg'  # Replace with the user's image path
identify_clothes(image_path)