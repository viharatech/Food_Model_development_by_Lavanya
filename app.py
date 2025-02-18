import flask
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import json

app = Flask(__name__)

# Model paths
MODEL_PATHS = {
    "Custom": "D:\\Lavanya\\Internship_task1\\custom_model.keras",
    "VGG16": "D:\\Lavanya\\Internship_task1\\vgg16.keras",
    "ResNet": "D:\\Lavanya\\Internship_task1\\resnet50_model.keras"
}

# Load models
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models[name] = load_model(path)
        except Exception as e:
            print(f"Error loading {name} model: {e}")
    else:
        print(f"Model file not found: {path}")


# Load JSON data safely
def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Load metrics and nutrition info
METRICS_PATHS = {
    "Custom": "D:\\Lavanya\\Internship_task1\\custom_model_metrics.json",
    "VGG16": "D:\\Lavanya\\Internship_task1\\vgg16_metrics.json",
    "ResNet": "D:\\Lavanya\\Internship_task1\\resnet_metrics.json"
}

metrics = {name: load_json(path) for name, path in METRICS_PATHS.items()}
NUTRITION_INFO_PATH = "D:\\Lavanya\\Internship_task1\\Food_information.json"
nutrition_info = load_json(NUTRITION_INFO_PATH)

# Class labels
CLASS_NAMES = [
    "apple_pie", "Baked potato", "burger", "butter_naan", "chai", "chapati",
    "cheesecake", "chicken curry", "chole_bhature", "Crispy Chicken", "dal_makhani",
    "dhokla", "Donut", "fried_rice", "Fries", "Hot Dog", "ice_cream", "idli",
    "jalebi", "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa", "momos",
    "omelette", "paani_puri", "pakode", "pav_bhaji", "Sandwich", "Taco", "Taquito"
]


# Preprocess image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/')
def home():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    print("Received request at /predict")  # Debug log

    if 'file' not in request.files:
        print("No file uploaded")  # Debug log
        return "No file uploaded", 400

    file = request.files['file']
    model_name = request.form.get('model')

    print(f"Selected Model: {model_name}")  # Debug log

    if model_name not in models:
        print("Invalid model selection")  # Debug log
        return "Invalid model selection"

    if file.filename == '':
        print("No selected file")  # Debug log
        return "No selected file"

    # Ensure 'uploads' directory exists
    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    print(f"File saved at: {file_path}")  # Debug log

    try:
        image = preprocess_image(file_path)
        print("Image preprocessing done")  # Debug log
    except Exception as e:
        print(f"Error in preprocessing: {e}")  # Debug log
        return str(e)

    # Make prediction
    try:
        predictions = models[model_name].predict(image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        print(f"Predicted class: {predicted_class}")  # Debug log

        class_metrics = metrics.get(model_name, {}).get(predicted_class, "No metrics available")
        class_nutrition = nutrition_info.get(predicted_class, "No nutrition info available")
    except Exception as e:
        print(f"Error in prediction: {e}")  # Debug log
        return str(e)

    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        predicted_class=predicted_class,
        class_metrics=class_metrics,
        class_nutrition=class_nutrition
    )


if __name__ == '__main__':
    app.run(debug=True)
