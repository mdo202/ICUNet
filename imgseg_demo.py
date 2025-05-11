import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
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
model = MobileNetV3_DeepLab(num_classes=1).to(device)
model_path = "models/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from checkpoint.")
else:
    print("No checkpoint found, starting from scratch.")
model.eval()

# OpenCV background subtraction setup
print("Capturing static background in 3 seconds...")
cap = cv2.VideoCapture(0)
cv2.waitKey(3000)
ret, background = cap.read()
background = cv2.resize(background, (512, 512))
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (512, 512))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    # ------------------------------
    # 1. Deep Learning Segmentation
    # ------------------------------
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    prob_map = torch.sigmoid(output)
    mask_dl = (prob_map.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    mask_rgb_dl = np.stack([mask_dl] * 3, axis=-1)
    foreground_dl = frame * mask_rgb_dl
    background_dl = np.full_like(frame, 128)
    result_dl = np.where(mask_rgb_dl == 1, foreground_dl, background_dl)

    # ---------------------------------------
    # 2. Traditional Background Subtraction
    # ---------------------------------------
    diff = cv2.absdiff(background_blur, frame_blur)
    _, mask_trad = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask_trad = cv2.dilate(mask_trad, kernel, iterations=2)
    mask_trad = cv2.morphologyEx(mask_trad, cv2.MORPH_CLOSE, kernel)
    mask_trad_clean = np.zeros_like(mask_trad)
    contours, _ = cv2.findContours(mask_trad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            cv2.drawContours(mask_trad_clean, [cnt], -1, 255, -1)
    mask_rgb_trad = np.stack([mask_trad_clean // 255] * 3, axis=-1)
    foreground_trad = frame * mask_rgb_trad
    background_trad = np.full_like(frame, 64)
    result_trad = np.where(mask_rgb_trad == 1, foreground_trad, background_trad)

    # -----------------------
    # Display all 3 side by side
    # -----------------------
    # Flip all images for mirrored view
    frame_flipped = cv2.flip(frame, 1)
    result_trad_flipped = cv2.flip(result_trad, 1)
    result_dl_flipped = cv2.flip(result_dl, 1)

    # Add labels to each frame
    cv2.putText(frame_flipped, "Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
    cv2.putText(result_trad_flipped, "Traditional", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
    cv2.putText(result_dl_flipped, "Trained Model", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

    # Stack all three images horizontally
    stacked = np.hstack((frame_flipped, result_trad_flipped, result_dl_flipped))
    cv2.imshow("Frame | Traditional | Trained Model", stacked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        print("Re-capturing background in 3 seconds...")
        cv2.waitKey(3000)
        ret, background = cap.read()
        background = cv2.resize(background, (512, 512))
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
