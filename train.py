from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(data="C:\\Users\\aditi\\OneDrive\\Desktop\\SwachhAlert\\data.yaml", epochs=30, imgsz=640)

# Define the path for saving the model (best.pt)
save_dir = './model_weights'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# The YOLO model automatically saves the best weights after training
# The best weights are stored in the 'runs' directory by default
best_model_path = os.path.join(save_dir, 'best.pt')

# Move the model weights from the default save location to the desired location
os.rename('runs/detect/train/weights/best.pt', best_model_path)

print(f"Model saved at {best_model_path}")
