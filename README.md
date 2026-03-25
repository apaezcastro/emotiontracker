# Emotion Detection App

[![Build and deploy container app to Azure Web App - andre-emotion-app](https://github.com/apaezcastro/emotiontracker/actions/workflows/main_andre-emotion-app.yml/badge.svg)](https://github.com/apaezcastro/emotiontracker/actions/workflows/main_andre-emotion-app.yml)

A computer vision app that detects emotions from a webcam feed. Built to learn how to deploy vision models on small compute. The knowledge directly informed how I approach model integration in TranslationXR.

**Honest limitation:** it runs on Azure but isn't truly real time for video. Face detection should have been handled on the frontend with WebSockets for backend communication. It's a project I learned more from than I expected.

---

## Stack

**Backend:** Python, Flask, Gunicorn, OpenCV, ONNXRuntime, NumPy
**Frontend:** HTML5, CSS3, JavaScript
**Deployment:** Docker, Azure App Service, GitHub Actions

---

## Run Locally (Docker)

**1. Clone the Repository**
```sh
git clone https://github.com/apaezcastro/emotiontracker.git
cd emotiontracker
```

**2. Download Model Files and place in `backend/`**
- `emotion-ferplus-8.onnx` from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)
- `haarcascade_frontalface_default.xml` from the [OpenCV repository](https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml)

**3. Build and Run**
```sh
cd backend
docker build -t emotion-app-backend .
docker run -p 5000:5000 emotion-app-backend
```

**4. Run the Frontend**
```sh
cd frontend
npm install
npm start
```

---

## Key Learnings
- Containerizing a Python app with Docker for consistent deployment
- Setting up a CI/CD pipeline with GitHub Actions for automated Azure deployments
- Deploying to Azure App Service
- Why frontend face detection with WebSockets would have been the better architecture

---

## Credits
Model: [emotion-ferplus-8.onnx](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus) from the ONNX Model Zoo
-----
