import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
import os
from deeplab import MobileNetV3_DeepLab

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Initialize the model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3_DeepLab(num_classes=1).to(device)  # Initialize the model
model_path = "models/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from checkpoint.")
else:
    print("No checkpoint found, starting from scratch.")
model.eval()  # Set the model to evaluation mode

# Start the webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (512, 512))
    if not ret:
        break

    # Preprocess the frame for model input
    input_tensor = preprocess(frame).unsqueeze(0).to(device)

    # Run the frame through the model to get the output
    with torch.no_grad():
        output = model(input_tensor)


    prob_map = torch.sigmoid(output)

    # Move to CPU and squeeze to 2D mask
    # These are all logits between 0 and 1
    mask = prob_map.squeeze().cpu().numpy()

    # Binarize the mask at threshold 0.5
    # This makes the mask binary (0 or 1)
    mask = (mask > 0.5).astype(np.uint8)

    # Ensure the mask has the same size as the frame
    assert mask.shape[:2] == frame.shape[:2], "Mask and frame must have the same size"

    # Convert to 3 channels for display (easier for color visualization)
    mask_rgb = np.stack([mask] * 3, axis=-1)
   
    # Use NumPy to apply the mask manually (instead of cv2.bitwise_and)
    foreground = frame * mask_rgb  # 0 or 1 mask for foreground
    background = cv2.GaussianBlur(frame, (27, 27), 0)
    background = np.full_like(frame, 128)

    # Combine the foreground with the blurred background
    result = np.where(mask_rgb == 1, foreground, background)
    # Display the result (foreground clear on blurred background)
    result = cv2.flip(result, 1)  
    cv2.imshow("Real-Time Segmentation", result)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
