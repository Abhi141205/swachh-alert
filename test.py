from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO("C:\\Users\\aditi\\OneDrive\\Desktop\\SwachhAlert\\runs\\detect\\train3\\weights\\best.pt")  # adjust path if needed

# Load and predict on a test image
image_path = "C:\\Users\\aditi\\OneDrive\\Desktop\\SwachhAlert\\test\\images\\download_jpeg.rf.f4e94dfdc55789f858d7ee321ec412b3.jpg"  # replace with the path to your test image
results = model.predict(image_path, show=True, save=True)

# Print class predictions
for box in results[0].boxes:
    cls = int(box.cls[0].item())
    conf = float(box.conf[0])
    print(f"Detected class {cls} with confidence {conf:.2f}")
