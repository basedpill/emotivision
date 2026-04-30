"""
emotiondetection web app

run:      python webapp/app.py
needs:    models/emotion_cnn.keras  (produced by train_model.py)
"""

import os, json, base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# paths
MODEL_PATH    = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_cnn.keras')
LABELS_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_labels.json')
HIST_PATH     = os.path.join(os.path.dirname(__file__), '..', 'models', 'training_history.json')
REPORT_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_report.json')
IMG_SIZE      = 48
MAX_IMG_BYTES = 5 * 1024 * 1024  # 5mb cap on each frame

EMOTION_META = {
    "angry":    {"emoji": "😠", "color": "#FF4500"},
    "disgust":  {"emoji": "🤢", "color": "#32CD32"},
    "fear":     {"emoji": "😨", "color": "#9370DB"},
    "happy":    {"emoji": "😄", "color": "#FFD700"},
    "sad":      {"emoji": "😢", "color": "#6495ED"},
    "surprise": {"emoji": "😲", "color": "#FF69B4"},
    "neutral":  {"emoji": "😐", "color": "#A9A9A9"},
}

print("[app] loading model...")
try:
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    CLASS_NAMES = [label_map[str(i)] for i in range(len(label_map))]
    print(f"[app] model loaded. classes: {CLASS_NAMES}")
except Exception as e:
    print(f"[error] {e}\n[info] run train_model.py first.")
    model = None
    CLASS_NAMES = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# build a two-output model once at startup so we don't recreate it on every request.
# target is the last conv2d layer (6x6x384 feature maps in block 4 for v5b).
last_conv_layer = None
grad_model      = None
if model is not None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
    if last_conv_layer is not None:
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model.output]
            )
            print(f"[app] grad-cam ready, target layer: {last_conv_layer.name}")
        except Exception as e:
            print(f"[app] grad-cam init failed: {e}")


# utilities

def decode_image(data_url):
    """decode a base64 data-url into an opencv bgr image array."""
    _, encoded = data_url.split(',', 1)
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def preprocess_face(roi):
    """convert a bgr face crop to a normalised (1, 48, 48, 1) float32 array."""
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (resized.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_emotion(face_tensor):
    """run the cnn and return emotions sorted by confidence (highest first)."""
    probs = model.predict(face_tensor, verbose=0)[0]
    results = [
        {
            "name":  CLASS_NAMES[i],
            "score": round(float(probs[i]) * 100, 1),
            "label": CLASS_NAMES[i].capitalize(),
            **EMOTION_META.get(CLASS_NAMES[i], {"emoji": "❓", "color": "#888"})
        }
        for i in range(len(CLASS_NAMES))
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_gradcam(tensor, class_idx):
    """compute a grad-cam heatmap for the given face tensor and emotion class.

    watches the input tensor so the tape records the full forward pass, then
    backprops to the last conv layer activations to get spatial importance.
    returns a 2d numpy array (h, w) normalised to [0, 1].
    """
    inp = tf.cast(tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        conv_out, preds = grad_model(inp, training=False)
        class_score = preds[0, class_idx]

    grads = tape.gradient(class_score, conv_out)  # shape: (1, 6, 6, channels)
    if grads is None:
        return np.zeros((6, 6))

    # average gradient magnitude across spatial+batch dims => importance per channel
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    cam     = conv_out[0].numpy() @ pooled
    cam     = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def make_heatmap_overlay(face_bgr, heatmap):
    """blend a grad-cam heatmap over a face crop. returns a base64 jpeg data url."""
    h, w      = face_bgr.shape[:2]
    hm        = cv2.resize(heatmap.astype(np.float32), (w, h))
    hm_color  = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)

    # convert face to grayscale->bgr so the jet overlay stands out clearly
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_rgb  = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(face_rgb, 0.45, hm_color, 0.55, 0)
    _, buf  = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


# routes

@app.route('/')
def index():
    return render_template('index.html', model_loaded=(model is not None))


@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({"success": False, "error": "model not loaded. run train_model.py first."}), 503

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "no image provided"}), 400

    if len(data['image']) > MAX_IMG_BYTES:
        return jsonify({"success": False, "error": "image too large"}), 413

    try:
        frame = decode_image(data['image'])
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"success": True, "faces": [], "face_count": 0})

        results = []
        for (x, y, w, h) in faces:
            pad = int(0.1 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            emotions = predict_emotion(preprocess_face(frame[y1:y2, x1:x2]))
            dom = emotions[0]
            results.append({
                "dominant": dom["name"],
                "dominant_config": {"emoji": dom["emoji"], "color": dom["color"], "label": dom["label"]},
                "emotions": emotions,
                "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

        return jsonify({"success": True, "faces": results, "face_count": len(results)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/gradcam', methods=['POST'])
def gradcam():
    """return grad-cam heatmap overlay images for each detected face."""
    if model is None or grad_model is None:
        return jsonify({"success": False, "error": "model not loaded"}), 503

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "no image"}), 400

    try:
        frame = decode_image(data['image'])
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"success": True, "heatmaps": []})

        heatmaps = []
        for (x, y, w, h) in faces:
            pad = int(0.1 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            tensor    = preprocess_face(face_crop)

            probs     = model.predict(tensor, verbose=0)[0]
            class_idx = int(np.argmax(probs))

            cam     = get_gradcam(tensor, class_idx)
            overlay = make_heatmap_overlay(face_crop, cam)

            heatmaps.append({
                "image":   overlay,
                "emotion": CLASS_NAMES[class_idx],
                "region":  {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

        return jsonify({"success": True, "heatmaps": heatmaps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "classes": CLASS_NAMES})


@app.route('/stats')
def stats():
    """return training stats so the frontend can populate the accuracy ring and per-class bars."""
    result = {
        "model_loaded":   model is not None,
        "test_accuracy":  None,
        "epochs_trained": None,
        "ensemble_size":  None,
        "class_report":   None,
    }

    # the canonical class_report.json is the v5b 3-model ensemble report.
    # if it has the 'ensemble_size' key we know the ensemble was used.
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            rpt = json.load(f)
        result["class_report"] = rpt
        # class_report_ensemble.json stores accuracy and ensemble metadata at the top level
        if isinstance(rpt, dict):
            if "accuracy" in rpt and isinstance(rpt["accuracy"], (int, float)):
                result["test_accuracy"] = float(rpt["accuracy"])
            if "ensemble_size" in rpt:
                result["ensemble_size"] = int(rpt["ensemble_size"])

    if os.path.exists(HIST_PATH):
        with open(HIST_PATH) as f:
            hist = json.load(f)
        # if test_accuracy wasn't set from the report, fall back to history
        if result["test_accuracy"] is None:
            result["test_accuracy"] = hist.get("test_accuracy_tta") or hist.get("test_accuracy")
        result["epochs_trained"] = len(hist.get("accuracy", []))

    return jsonify(result)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print("emotiondetection starting on http://localhost:5000")
    app.run(debug=debug, host='0.0.0.0', port=5000)
