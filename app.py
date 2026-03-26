import base64
import numpy as np
import cv2
import onnxruntime as ort
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion-tracker'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

session = ort.InferenceSession('emotion-ferplus-8-quant.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(img_gray):
    img = cv2.resize(img_gray, (64, 64))
    return img.astype(np.float32)[np.newaxis, np.newaxis, :, :]  # [1,1,64,64]


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('frame')
def handle_frame(data):
    try:
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        inp = preprocess(img)
        logits = session.run(None, {input_name: inp})[0][0]
        emotion = EMOTIONS[int(np.argmax(softmax(logits)))]
        emit('emotion', {'emotion': emotion})
    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)
