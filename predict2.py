from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('best.pt')

# Perform object detection and get predictions
results = model.predict(source='shuttle_19.jpg', show=False, conf=0.5, save=False)

# Retrieve bounding box coordinates of objects labeled as 'net_level'
net_level_objects = results.filter('net_level').xyxy[0].numpy().tolist()

# Print bounding box coordinates for 'net_level' objects
for bbox in net_level_objects:
    class_id, confidence, x_min, y_min, x_max, y_max = bbox[:6]  # Extracting coordinates

    # Bounding box coordinates
    bounding_box_coordinates = {
        "class_id": class_id,
        "confidence": confidence,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max
    }

    print("Bounding Box Coordinates for 'net_level' objects:", bounding_box_coordinates)
