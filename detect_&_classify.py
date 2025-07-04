import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet18
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image

# Step 1: Load Faster R-CNN model for detection
detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()  # Set to evaluation mode

# Step 2: Load ResNet model for classification
classification_model = resnet18(pretrained=True)
classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, 3)  # 3 classes: Men, Women, Child
classification_model.load_state_dict(torch.load("clothing_classifier.pth"))  # Load the trained model
classification_model.eval()  # Set to evaluation mode

# Step 3: Preprocessing transformations for classification
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 4: Detect clothing in the frame
def detect_clothing(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_image = F.to_tensor(image)

    with torch.no_grad():
        predictions = model([tensor_image])

    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']

    # Filter predictions with a confidence score > 0.5
    filtered_boxes = [box for i, box in enumerate(boxes) if scores[i] > 0.5]
    return filtered_boxes

# Step 5: Classify the cropped clothing regions
def classify_clothing(cropped_image, model):
    input_tensor = preprocess(cropped_image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ["Men", "Women", "Child"]
    return classes[predicted.item()]

# Step 6: Open webcam and process frames
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    boxes = detect_clothing(frame, detection_model)

    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
        cropped_image = frame[y1:y2, x1:x2]  # Crop the detected region
        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Classify the cropped region
        label = classify_clothing(cropped_pil_image, classification_model)

        # Draw the bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Clothing Detection and Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
