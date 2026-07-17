"""
Flask API for the Butterfly Image Classification model (EfficientNetB0 + PyTorch).

Run with:
    python app.py

Then:
    - Open http://127.0.0.1:5000 in a browser for a simple upload UI
    - Or POST an image to /predict, e.g.:
        curl -X POST -F "file=@image.jpg" "http://127.0.0.1:5000/predict?top_k=5"
"""

import io
import json
import os

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config & artifacts
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH       = os.path.join(BASE_DIR, "best_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
CONFIG_PATH      = os.path.join(BASE_DIR, "config.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# Fall back to the training defaults if config.json wasn't generated
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
else:
    config = {"img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

IMG_SIZE = config["img_size"]
MEAN     = config["mean"]
STD      = config["std"]

# ---------------------------------------------------------------------------
# Build model architecture (must match training) and load weights
# ---------------------------------------------------------------------------
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features  # 1280

model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes),
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------------------------------------------------------------------
# Preprocessing (same as predict_image() in the notebook)
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


def predict_image_bytes(image_bytes, top_k=5):
    """Run the model on raw image bytes and return top-k (label, probability) pairs."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    top_k = max(1, min(top_k, num_classes))
    top_probs, top_idx = torch.topk(probs, k=top_k)

    return [
        {"class": class_names[idx], "confidence": round(float(prob), 6)}
        for prob, idx in zip(top_probs, top_idx)
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        top_k = int(request.args.get("top_k", 5))
    except ValueError:
        top_k = 5

    try:
        image_bytes = file.read()
        results = predict_image_bytes(image_bytes, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    return jsonify({
        "filename": file.filename,
        "predicted_class": results[0]["class"],
        "confidence": results[0]["confidence"],
        "top_k": results,
    })


if __name__ == "__main__":
    # Set debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
