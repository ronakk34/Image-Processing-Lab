import cv2

# Open the webcam (0 for default camera)
# Note: Direct webcam access usually doesn't work in Google Colab as it runs on remote servers.
# This code is provided for environments where webcam access is possible.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    # In Colab, you might need to upload a video file and process it instead.
    exit()

print("Press 'q' to exit.") # Corrected string literal

while True:
    # Read a frame
    ret, frame = cap.read() # Added assignment operator

    if not ret:
        print("Failed to capture frame")
        break

    # Real-Time Processing
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Corrected assignment and constant

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200) # Added assignment operator and correct function call

    # Display processed frame
    cv2.imshow("Live Webcam Edges", edges)

    # Press 'q' to quit (consistent with print statement)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Corrected quit key check
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()