import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
import base64
import cv2
from flask import send_from_directory
app = Flask(__name__)

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load ONNX model
session = ort.InferenceSession("emotion-ferplus-8.onnx", providers=["CPUExecutionProvider"])

emotion_table = {
    0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
    4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'
}

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Detect face(s)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None  # No face detected

    # Use the first detected face
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    # Resize face to model input size
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype(np.float32)
    face_img = face_img[np.newaxis, np.newaxis, :, :]  # shape: (1, 1, 64, 64)
    return face_img

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        img_data = base64.b64decode(data['image'].split(",")[1])
        with open("static/debug_frame.jpg", "wb") as f:
            f.write(img_data)

        input_img = preprocess_image(img_data)
        if input_img is None:
            return jsonify({'error': 'Invalid image'}), 400
        debug_img = np.squeeze(input_img) * 255  # (64, 64) grayscale image
        debug_img = debug_img.astype(np.uint8)
        cv2.imwrite("static/debug_preprocessed.jpg", debug_img)
        inputs = {session.get_inputs()[0].name: input_img}
        raw_output = session.run(None, inputs)[0]
        print(raw_output)
        probs = softmax(raw_output)
        top_idx = np.argmax(probs)
        print(f"here is top_idx: {top_idx}")
        print(emotion_table)
        print(probs)
        return jsonify({'emotion': emotion_table[top_idx],'confidence': float(probs[0][top_idx]),'raw_probabilities': probs[0].tolist()})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Detection failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
