# Emotion Detection App

[![Build and deploy container app to Azure Web App - andre-emotion-app](https://github.com/apaezcastro/emotiontracker/actions/workflows/main_andre-emotion-app.yml/badge.svg)](https://github.com/apaezcastro/emotiontracker/actions/workflows/main_andre-emotion-app.yml)

A computer vision app that detects emotions from a webcam feed. Built to learn how to deploy vision models on small compute. The knowledge directly informed how I approach model integration in TranslationXR.

---

## Stack

**Backend:** Python, Flask-SocketIO, Gunicorn (eventlet), OpenCV, ONNXRuntime, NumPy
**Frontend:** HTML5, CSS3, JavaScript, face-api.js, Socket.io
**Deployment:** Docker, Azure App Service, GitHub Actions

---

## Run Locally (Docker)

**1. Clone the Repository**
```sh
git clone https://github.com/apaezcastro/emotiontracker.git
cd emotiontracker
```

**2. Build and Run**
```sh
docker build -t emotion-app-backend .
docker run -p 8000:8000 emotion-app-backend
```

**3. Open in browser**
```
http://localhost:8000
```

---

## Known Limitation
FER+ was trained on posed, exaggerated expressions. Subtle emotions like disgust, contempt, and fear are unreliable in natural use. Fine-tuning on a real-world dataset like AffectNet is the next improvement.

---

## Key Learnings
- Containerizing a Python app with Docker for consistent deployment
- Setting up a CI/CD pipeline with GitHub Actions for automated Azure deployments
- Deploying to Azure App Service
- Moving face detection to the frontend with WebSockets dramatically reduces server load and latency

---

## Credits
Model: [emotion-ferplus-8.onnx](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus) from the ONNX Model Zoo
-----
