import cv2
import numpy as np
import os
import time

# Function to adjust contrast and brightness of a frame
def change_contrast_brightness(frame, alpha, beta):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Function to capture the background.
# Applies contrast and brightness adjustments to the background image.
# This is used to compare with the current frame to detect the person.
def capture_background(cap, contrast_brightness_combos):
    print("Please step out of the frame. Capturing background in 3 seconds...")
    cv2.waitKey(3000)
    ret, background = cap.read()
    background = cv2.flip(background, 1)
    background = cv2.resize(background, (512, 512))
    background_og = background.copy()

    backgrounds = []
    for alpha, beta in contrast_brightness_combos:
        background_adj = change_contrast_brightness(background_og, alpha, beta)
        background_gray = cv2.cvtColor(background_adj, cv2.COLOR_BGR2GRAY)
        backgrounds.append(background_gray)

    return background_og, backgrounds

if __name__ == "__main__":
    contrast_brightness_combos = [(1, 0),
                                  (0.5, 0), (0.75, 0), (1.5, 0), (2, 0),
                                  (1, -100), (1, -200), (1, -300),
                                  (1, 100), (1, 150), (1, 200),
                                  (-1.5, 150), (-2, 200), (1.5, -150), (2, -200)
                                  ]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(0)

    frame_interval = 0.1  # seconds
    last_saved_time = time.time()

    output_dir = "data"
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    background_og, backgrounds = capture_background(cap, contrast_brightness_combos)

    recording = False
    delay_passed = False
    record_start_time = None
    frame_count = 0

    try:
        while True:
            # Read frame from webcam
            # Applies contrast and brightness adjustments to the current frame.
            # Other preprocessing steps are also applied to the current frame
            # identifies contours of a person, and combines contours from different contrast/brightness adjustments.
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (512, 512))
            frame_og = frame.copy()

            person_masks = []
            for i in range(len(contrast_brightness_combos)):
                alpha, beta = contrast_brightness_combos[i]
                frame_adj = change_contrast_brightness(frame_og, alpha, beta)
                frame_gray = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(backgrounds[i], frame_gray)
                _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
                mask = cv2.dilate(mask, kernel, iterations=3)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mask_clean = np.zeros_like(mask)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 5000:
                        cv2.drawContours(mask_clean, [cnt], -1, 255, -1)
                person_masks.append(mask_clean)

            # Aggregate using majority voting
            stacked_masks = np.stack(person_masks, axis=-1)
            votes = np.sum(stacked_masks == 255, axis=-1)
            majority_threshold = len(contrast_brightness_combos) // 2
            final_mask = np.where(votes > majority_threshold, 255, 0).astype(np.uint8)

            # Foreground and background separation
            # Applies the final combined mask (person) to the original frame.
            # Whatever was detected as a person is kept, and the rest of the frame is filled with a solid color.
            person = cv2.bitwise_and(frame_og, frame_og, mask=final_mask)
            background_edited = np.full_like(frame_og, (0, 140, 255))
            inverted_mask = cv2.bitwise_not(final_mask)
            background_only = cv2.bitwise_and(background_edited, background_edited, mask=inverted_mask)
            final_combined = cv2.add(person, background_only)
            cv2.imshow("Detection", final_combined)

            key = cv2.waitKey(1) & 0xFF
            
            # Press r to record, press n to reset background, and ESC to exit
            if key == ord('r'):
                if not recording:
                    recording = True
                    delay_passed = False
                    record_start_time = time.time()
                    print("Recording will start in 3 seconds...")
                else:
                    recording = False
                    delay_passed = False
                    print("Recording stopped.")

            elif key == ord('n'):
                background_og, backgrounds = capture_background(cap, contrast_brightness_combos)
                print("Background reset.")
                recording = False
                delay_passed = False

            elif key == 27:  # ESC
                break

            if recording:
                if not delay_passed:
                    if time.time() - record_start_time >= 3:
                        delay_passed = True
                        last_saved_time = time.time()  # reset so it doesn't immediately save
                        print("Recording started.")
                else:
                    now = time.time()
                    if now - last_saved_time >= frame_interval:
                        timestamp = time.time()
                        filename = f"{int(timestamp * 1000)}.png"
                        cv2.imwrite(os.path.join(output_dir, "frames", filename), frame_og)
                        cv2.imwrite(os.path.join(output_dir, "masks", filename), final_mask)
                        # print(f"Saved: {filename}")
                        last_saved_time = now
                        frame_count += 1

    finally: 
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames saved: {frame_count}")
