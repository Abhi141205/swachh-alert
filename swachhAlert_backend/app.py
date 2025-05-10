from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# ✅ Load your trained YOLOv8 model (ensure the path is correct)
model = YOLO(r"C:\Users\aditi\OneDrive\Desktop\SwachhAlert\runs\detect\train3\weights\best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # ✅ Run prediction
    results = model(image, conf=0.2)  # Confidence threshold set to 0.2
    names = model.names  # Class names from model
    output = []

    # ✅ Extract detections
    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            conf = float(box[4])
            if conf > 0.2:
                cls = int(box[5])  # Class ID
                x_center, y_center, width, height = box[:4]

                # Convert xywh → xyxy
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                output.append({
                    "class": names[cls],
                    "confidence": round(conf, 2),
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                })

    return jsonify({"prediction": output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
