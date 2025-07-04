import cv2
import torch
import torch.nn as nn  # ✅ Import fixed
from torchvision import transforms, models
from PIL import Image

# Load the trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # Replace 3 with the number of classes
model.load_state_dict(torch.load('clothes_model.pth', map_location=torch.device('cpu')))  # map to CPU if needed
model.eval()

# Class names
class_names = ['Shirt', 'Jacket', 'Pants']  # ✅ Ensure these match your trained classes

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_clothing(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# Camera capture
cap = cv2.VideoCapture(0)
print("Press 'q' to exit the camera.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict clothes
    prediction = predict_clothing(frame, model)

    # Display the frame with prediction
    cv2.putText(frame, f"Detected: {prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Clothing Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
