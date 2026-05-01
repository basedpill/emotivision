"""
preprocessing utilities for the emotiondetection webapp.

extracted into its own module so that:
  - the same code is used at training time (via flow_from_directory rescaling)
    and at inference time (via preprocess_face below). centralising the 48x48
    greyscale + [0,1] normalisation prevents silent train/inference divergence.
  - the unit tests can import preprocess_face and decode_image directly,
    without loading the trained model weights.

if you change the input shape, dtype, or normalisation here, you almost
certainly also need to retrain the model. see chapter 7 reflection on why.
"""

import base64
import numpy as np
import cv2

# config (kept here, not in app.py, so tests can import them too)
IMG_SIZE      = 48
MAX_IMG_BYTES = 5 * 1024 * 1024  # 5 MB cap on each frame

EMOTION_META = {
    "angry":    {"emoji": "😠", "color": "#FF4500"},
    "disgust":  {"emoji": "🤢", "color": "#32CD32"},
    "fear":     {"emoji": "😨", "color": "#9370DB"},
    "happy":    {"emoji": "😄", "color": "#FFD700"},
    "sad":      {"emoji": "😢", "color": "#6495ED"},
    "surprise": {"emoji": "😲", "color": "#FF69B4"},
    "neutral":  {"emoji": "😐", "color": "#A9A9A9"},
}


def decode_image(data_url):
    """decode a base64 data-url into an opencv bgr image array.

    raises a ValueError if the payload is malformed or cannot be decoded.
    """
    if ',' not in data_url:
        raise ValueError("expected a base64 data url with a comma separator")
    _, encoded = data_url.split(',', 1)
    try:
        raw = base64.b64decode(encoded, validate=False)
    except Exception as e:
        raise ValueError(f"base64 decode failed: {e}")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("opencv could not decode the image bytes")
    return img


def preprocess_face(roi):
    """convert a bgr face crop to a normalised (1, 48, 48, 1) float32 array.

    matches the training-time pipeline: greyscale, resize to 48x48 with bilinear
    interpolation, scale to [0, 1], add the batch and channel axes.
    """
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (resized.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)