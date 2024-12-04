import cv2
import numpy as np
import easyocr

# Load the cascade for number plate detection
plate_cascade = cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')

# Create an EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language you want to use

# Open the video file
cap = cv2.VideoCapture('truck_video.mp4')

# Open a text file to write detected number plates
with open("detected_numberplates.txt", "w") as file:
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for yellow color in HSV
        lower_yellow = np.array([20, 100, 100])  # Adjust this as needed
        upper_yellow = np.array([30, 255, 255])  # Adjust this as needed

        # Create a mask for yellow color
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Optional: Perform morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of the yellow areas
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter contours based on area to find potential number plates
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle around the detected yellow region
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Use the detected yellow area to check for plates
                plate_region = frame[y:y + h, x:x + w]
                gray_plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

                # Use EasyOCR to read text from the plate region
                results = reader.readtext(plate_region)

                for (bbox, text, prob) in results:
                    if prob > 0.5:  # Confidence threshold
                        print(f"Detected Plate: {text}")
                        file.write(text + "\n")  # Write the detected plate to the file

                        # Draw bounding box and text on the frame
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        top_left = tuple(map(int, top_left))
                        bottom_right = tuple(map(int, bottom_right))
                        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detected yellow number plates
        cv2.imshow('Yellow Number Plate Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
