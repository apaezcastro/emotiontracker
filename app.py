from flask import Flask, render_template, request, jsonify
from fer import FER
import cv2
import numpy as np
import base64

app = Flask(__name__)
emotion_detector = FER(mtcnn=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'emotion': 'No image data'}), 400

    try:
        img_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Image decode failed: {e}")
        return jsonify({'emotion': 'Decode error'})

    if img is None:
        return jsonify({'emotion': 'Invalid image'})

    try:
        emotion, score = emotion_detector.top_emotion(img)
        if emotion is None:
            return jsonify({'emotion': 'No face'})
        return jsonify({'emotion': emotion})
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        return jsonify({'emotion': 'Detection error'})

if __name__ == '__main__':
    app.run(debug=True)
