"""
tests for the flask routes and preprocessing utilities.

run with:  pytest tests/
"""
import sys
import os
import base64
import numpy as np
import pytest
import unittest.mock as mock

# make webapp importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'webapp'))

# patch keras before importing app so we don't load the real model on every test run.
# _fake_preds has 7 values matching the 7 emotion classes; happy (index 3) is dominant.
_fake_preds = np.array([[0.10, 0.05, 0.05, 0.60, 0.05, 0.10, 0.05]])
_mock_model  = mock.MagicMock()
_mock_model.predict.return_value = _fake_preds

with mock.patch('tensorflow.keras.models.load_model', return_value=_mock_model):
    import app as ev_app
    from preprocess import preprocess_face, decode_image, EMOTION_META


@pytest.fixture
def client():
    ev_app.app.config['TESTING'] = True
    with ev_app.app.test_client() as c:
        yield c


# preprocess_face

def test_preprocess_face_output_shape():
    roi = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    out = preprocess_face(roi)
    assert out.shape == (1, 48, 48, 1)


def test_preprocess_face_dtype():
    roi = np.ones((60, 60, 3), dtype=np.uint8) * 200
    out = preprocess_face(roi)
    assert out.dtype == np.float32


def test_preprocess_face_normalised():
    # all-white crop should produce values clamped to [0, 1]
    roi = np.ones((64, 64, 3), dtype=np.uint8) * 255
    out = preprocess_face(roi)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_preprocess_face_tiny_crop():
    # crops smaller than 48x48 should be resized without crashing
    roi = np.random.randint(0, 255, (15, 15, 3), dtype=np.uint8)
    out = preprocess_face(roi)
    assert out.shape == (1, 48, 48, 1)


# decode_image

def test_decode_image_returns_bgr_array():
    import cv2
    dummy = np.zeros((48, 48, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', dummy)
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
    img = decode_image(b64)
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3   # bgr


# /health

def test_health_returns_200(client):
    assert client.get('/health').status_code == 200


def test_health_has_required_keys(client):
    data = client.get('/health').get_json()
    assert 'status'       in data
    assert 'model_loaded' in data
    assert 'classes'      in data


def test_health_seven_classes(client):
    data = client.get('/health').get_json()
    assert len(data['classes']) == 7


# / (index page)

def test_index_returns_html(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'EmotionDetection' in resp.data


# /stats

def test_stats_returns_200(client):
    assert client.get('/stats').status_code == 200


def test_stats_has_expected_keys(client):
    data = client.get('/stats').get_json()
    assert 'model_loaded'   in data
    assert 'test_accuracy'  in data
    assert 'epochs_trained' in data


def test_stats_accuracy_in_range(client):
    data = client.get('/stats').get_json()
    if data['test_accuracy'] is not None:
        assert 0.0 < data['test_accuracy'] < 1.0


# /analyze

def test_analyze_empty_body(client):
    resp = client.post('/analyze', data='', content_type='application/json')
    assert resp.status_code in (400, 503)


def test_analyze_missing_image_key(client):
    resp = client.post('/analyze', json={'other': 'x'})
    assert resp.status_code in (400, 503)


def test_analyze_503_when_model_none(client):
    saved = ev_app.model
    ev_app.model = None
    resp = client.post('/analyze', json={'image': 'data:image/jpeg;base64,abc'})
    ev_app.model = saved
    assert resp.status_code == 503


def test_analyze_blank_image_returns_no_faces(client):
    """a plain black frame shouldn't trigger face detection."""
    import cv2
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank)
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
    resp = client.post('/analyze', json={'image': b64})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert data['face_count'] == 0


# /gradcam

def test_gradcam_get_returns_405(client):
    """grad-cam only accepts POST; GET should return method-not-allowed."""
    resp = client.get('/gradcam')
    assert resp.status_code == 405

def test_gradcam_no_body(client):
    resp = client.post('/gradcam', data='', content_type='application/json')
    assert resp.status_code in (400, 503)


def test_gradcam_missing_image_key(client):
    resp = client.post('/gradcam', json={'other': 'x'})
    assert resp.status_code in (400, 503)


def test_gradcam_blank_image_no_faces(client):
    """a plain black image should return success=True with no heatmaps."""
    import cv2
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank)
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
    resp = client.post('/gradcam', json={'image': b64})
    # if grad_model is None (mocked), 503; if loaded, 200 with no heatmaps
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.get_json()
        assert data['success'] is True
        assert isinstance(data['heatmaps'], list)


# EMOTION_META

def test_all_seven_emotions_present():
    expected = {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}
    assert expected == set(EMOTION_META.keys())


def test_each_emotion_has_color_and_emoji():
    for name, meta in EMOTION_META.items():
        assert 'color' in meta, f"missing color for {name}"
        assert 'emoji' in meta, f"missing emoji for {name}"

def test_analyze_oversized_payload(client):
    """payloads above the 5 MB cap should return 413."""
    big = 'data:image/jpeg;base64,' + ('A' * (6 * 1024 * 1024))
    resp = client.post('/analyze', json={'image': big})
    assert resp.status_code in (413, 503)


def test_analyze_truncated_base64(client):
    """truncated base64 should be rejected, not crash."""
    resp = client.post('/analyze', json={'image': 'data:image/jpeg;base64,!!!notb64!!!'})
    assert resp.status_code in (400, 500, 503)