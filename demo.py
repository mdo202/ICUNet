import cv2
import numpy as np
import os

################################################################
### DEMO SCRIPT to show contrast / brightness adjustments 
################################################################

def change_brightness(frame, alpha, beta):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

if __name__ == "__main__":

    contrast_brightness_combos = [(1, 0), (-1, 0),
                                  (0.5, 0), (2, 0),
                                  (1, -100), (1, 100)]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap = cv2.VideoCapture(0)
    print("Please step out of the frame. Capturing background in 3 seconds...")

    def capture_background():
        print("Capturing background in 3 seconds...")
        cv2.waitKey(3000)
        ret, background = cap.read()
        background = cv2.flip(background, 1)
        background = cv2.resize(background, (512, 512))
        background_og = background.copy()

        backgrounds.clear()  # Clear previous backgrounds
        for alpha, beta in contrast_brightness_combos:
            background_adj = change_brightness(background_og, alpha, beta)
            background_gray = cv2.cvtColor(background_adj, cv2.COLOR_BGR2GRAY)
            background_gray_eq = clahe.apply(background_gray)
            backgrounds.append(background_gray_eq)
        print("Background capture complete.")

    # Initial background capture
    backgrounds = []
    capture_background()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (512, 512))
        frame_og = frame.copy()

        person_masks = []

        # Find masks for each contrast/brightness adjustment, by contour detection
        for i in range(len(contrast_brightness_combos)):
            alpha, beta = contrast_brightness_combos[i]
            frame_adj = change_brightness(frame_og, alpha, beta)
            frame_gray = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2GRAY)
            frame_gray_eq = clahe.apply(frame_gray)
            diff = cv2.absdiff(backgrounds[i], frame_gray_eq)
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

        # Apply final mask to create combined frame
        person = cv2.bitwise_and(frame_og, frame_og, mask=final_mask)
        background_edited = np.full_like(frame_og, (0, 140, 255))
        inverted_mask = cv2.bitwise_not(final_mask)
        background_only = cv2.bitwise_and(background_edited, background_edited, mask=inverted_mask)
        final_combined = cv2.add(person, background_only)

        # Display individual person masks in separate windows
        for i, mask in enumerate(person_masks):
            contour, brightness = contrast_brightness_combos[i]
            # Overlay person on original frame using the mask
            person_view = cv2.bitwise_and(frame_og, frame_og, mask=mask)
            
            labeled = person_view.copy()
            cv2.putText(labeled, f"Contrast {contour}, Brightness {brightness}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow(f"Contrast {contour}, Brightness {brightness}", labeled)

        # Also show the majority-voted final result
        combined_labeled = final_combined.copy()
        cv2.putText(combined_labeled, "Majority Voted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Majority Voted", combined_labeled)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            capture_background()
            print("Background reset.")

        elif key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
