# from ultralytics import YOLO

# model = YOLO('best.pt')

# model.predict(source='MVI_9716.MP4_Rendered_002.mp4', show=True, conf=0.5, save=True)

from ultralytics import YOLO
import cv2

# Initialize the YOLO model
model = YOLO('best.pt')

# Perform object detection and get predictions
# results = model.predict(source='MVI_9716.MP4_Rendered_002.mp4', show=True, conf=0.5, save=True)

# shuttle_19.jpg
results = model.predict(source='shuttle_19.jpg', show=True, conf=0.5, save=True)


# Filter out only the "shuttle" class predictions
shuttle_predictions = results.filter(class_name='shuttle')

print(shuttle_predictions)

# Process predictions to get object coordinates
# for prediction in results.xyxy[0]:
#     class_id, confidence, x_min, y_min, x_max, y_max = prediction.tolist()
    
#     # Calculate bounding box center coordinates
#     x_center = (x_min + x_max) / 2
#     y_center = (y_min + y_max) / 2
    
#     # Width and height of the bounding box
#     box_width = x_max - x_min
#     box_height = y_max - y_min
    
#     # Print or use coordinates as needed
#     print(f"Class ID: {class_id}, Confidence: {confidence}, Center: ({x_center}, {y_center}), Width: {box_width}, Height: {box_height}")
