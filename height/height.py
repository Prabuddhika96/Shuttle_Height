from models.experimental import attempt_load
import torch
import cv2

# Load the model
model = attempt_load('../best.pt', map_location=torch.device('cpu')).autoshape()

# Load the video
cap = cv2.VideoCapture('../MVI_9716.MP4_Rendered_001.mp4')

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Access bounding box coordinates of detected objects
    for pred in results.pred:
        # Extracting coordinates (x_center, y_center, width, height)
        x_center, y_center, width, height = pred[:4]  # Normalized coordinates

        # Convert to absolute coordinates (pixel values)
        abs_x = x_center * frame.shape[1]
        abs_y = y_center * frame.shape[0]
        abs_width = width * frame.shape[1]
        abs_height = height * frame.shape[0]

        # Calculate bounding box corners
        x1 = int(abs_x - abs_width / 2)
        y1 = int(abs_y - abs_height / 2)
        x2 = int(abs_x + abs_width / 2)
        y2 = int(abs_y + abs_height / 2)

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frame with bounding boxes
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
