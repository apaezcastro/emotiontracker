# Real-Time Emotion and Face Detection App

[![Build and deploy container app to Azure Web App - andre-emotion-app](https://github.com/apaezcastro/emotiontracker/actions/workflows/main_andre-emotion-app.yml/badge.svg)](https://github.com/apaezcastro/emotiontracker/actions/workflows/main_andre-emotion-app.yml)

A full-stack web application that performs real-time emotion detection and face tracking from a live webcam feed. This app uses a Python backend for AI inference and a JavaScript frontend to render the results dynamically on an HTML5 canvas.

**Live Demo:** [andre-emotion-app.azurewebsites.net](https://andre-emotion-app.azurewebsites.net)

---

## Features

* **Live Emotion Detection:** Identifies the user's emotion (e.g., Happy, Sad, Neutral) in real-time.
* **Dynamic Bounding Box:** Draws a precise box around the detected face on the video feed.
* **Responsive UI:** The emotion label is anchored to the bounding box and moves with it.
* **Robust Concurrency:** The app is built with a client-side "lock" to ensure stable, real-time performance by processing one frame at a time, preventing request stacking.

---

## Tech Stack

### Backend
* **Python:** Core application language.
* **Flask:** Serves the API endpoint.
* **Gunicorn:** Production web server (used in Docker).
* **OpenCV (`opencv-python-headless`):** For image processing (`imdecode`) and face detection (Haar Cascades).
* **ONNXRuntime:** To run the `emotion-ferplus-8.onnx` model.
* **NumPy:** For high-performance array manipulation.
* **Flask-Cors:** To handle cross-origin requests from the frontend.

### Frontend
* **HTML5:** `video` and `canvas` elements for video streaming.
* **CSS3:** Styling for the video, bounding box, and overlay.
* **JavaScript (ES6+):** Manages the `setInterval` loop, captures frames, and uses the `fetch` API.

### Deployment
* **Docker:** Containerizes the Python backend for consistent, isolated deployment.
* **GitHub:** Source code repository.
* **Azure (CI/CD):** Automated build and deployment pipeline (via GitHub Actions) to an Azure App Service.

---

## 🚀 Getting Started (Running Locally)

There are two ways to run this project locally. Docker is the recommended method as it handles all dependencies automatically.

### Prerequisites

* [Git](https://git-scm.com/downloads)
* [Node.js (which includes npm)](https://nodejs.org/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for the Docker method)
* [Python 3.8+](https://www.python.org/downloads/) (for the manual Python method)

---

### Option 1: Running with Docker (Recommended)

This method builds the backend in a container, so you do not need to install Python or any packages on your local machine.

**1. Clone the Repository**
```sh
git clone [https://github.com/apaezcastro/emotiontracker.git](https://github.com/apaezcastro/emotiontracker.git)
cd emotiontracker
````

**2. Download Model Files**

The Docker build expects the model files to be present.

  * **Emotion Model (`emotion-ferplus-8.onnx`):**
    Download it from the ONNX Model Zoo [here](https://www.google.com/search?q=https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx).
    Place this file in the `backend/` directory.

  * **Face Detection Model (`haarcascade_frontalface_default.xml`):**
    Download it from the OpenCV repository [here](https://www.google.com/search?q=https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml).
    Place this file in the `backend/` directory.

**3. Build and Run the Backend Container**

```sh
# Navigate to the backend directory
cd backend

# Build the Docker image
docker build -t emotion-app-backend .

# Run the container, mapping port 5000
docker run -p 5000:5000 emotion-app-backend
```

The backend is now running on `http://localhost:5000`.

**4. Run the Frontend**

In a **new terminal**, navigate to the frontend directory:

```sh
# (From the root directory)
cd frontend

# Install npm packages
npm install

# Run the development server
npm start
```

Your browser will open to `http://localhost:3000`, which will connect to the backend container running on port 5000.

-----

### Option 2: Running Manually (Python & Node.js)

This method requires you to set up both the Python and Node.js environments manually.

**1. Clone the Repository**

```sh
git clone [https://github.com/apaezcastro/emotiontracker.git](https://github.com/apaezcastro/emotiontracker.git)
cd emotiontracker
```

**2. Download Model Files**

  * **Emotion Model (`emotion-ferplus-8.onnx`):**
    Download it from the ONNX Model Zoo [here](https://www.google.com/search?q=https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx).
    Place this file in the `backend/` directory.

  * **Face Detection Model (`haarcascade_frontalface_default.xml`):**
    Download it from the OpenCV repository [here](https://www.google.com/search?q=https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml).
    Place this file in the `backend/` directory.

**3. Backend Setup (Python)**

```sh
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

# Run the Flask server
# (It will typically run on http://localhost:5000)
python app.py
```

**4. Frontend Setup (JavaScript)**

In a **new terminal**:

```sh
# Navigate to the frontend directory
cd frontend

# Install npm packages
npm install

# Run the development server
npm start
```

**5. View Your App**

Open your browser and go to `http://localhost:3000`.

-----

## How It Works

1.  The JavaScript frontend uses `setInterval` (set to `1000ms` for production) to capture a frame from the user's webcam.
2.  The frame is drawn to an HTML `canvas` and exported as a `image/jpeg` base64 data URL.
3.  This data is sent via a `fetch` POST request to the `/detect_emotion` API endpoint.
4.  A **concurrency lock** (`analyzing` flag) in JavaScript ensures that if the server *does* take longer than the interval, a new request is not sent until the previous one completes.
5.  The Python backend receives the JSON, decodes the base64 string into a NumPy array (`nparr`), and then uses `cv2.imdecode` to create an image.
6.  The AI model runs, returning the emotion and a 1D array for the face coordinates: `[x, y, width, height]`.
7.  The Python backend converts the NumPy array to a standard Python list (`.tolist()`) and returns the emotion and coordinates as JSON.
8.  The frontend receives the JSON, clears any old bounding boxes, and draws a new box and emotion overlay using the received coordinates.

-----

## Design Choices & Key Learnings

This project involved several key design decisions to achieve a stable, real-time feel.

  * **Concurrency Lock (`analyzing` flag):** This is the most critical design choice. It prevents "request stacking" and makes the app resilient to performance differences between servers.

      * **Locally:** On a 12-core machine, the detection runs in `~100ms`, allowing for a 10 FPS loop.
      * **On Azure (Shared Tier):** The same detection takes `~700-900ms`. The `analyzing` flag gracefully handles this, ensuring the app remains stable and simply updates less frequently, rather than crashing.

  * **Lightweight Data Transfer:** The backend only sends back tiny JSON packets (e.g., `~400B`) containing the emotion string and four coordinates. This ensures that the bottleneck is *always* CPU processing time, not network lag.

  * **NumPy to List (`.tolist()`):** A `numpy.ndarray` is not JSON serializable. The `.tolist()` method was essential to convert the detection results into a standard Python list that Flask/FastAPI can serialize into JSON.

  * **Single Face (1D Array):** The app is currently optimized to detect the single most prominent face, returning a simple 1D array `[x, y, w, h]`. This simplified the frontend logic, though it's a target for future improvement.

  * **Frontend Error Handling:** The frontend JavaScript checks if `faceData` exists before trying to draw a box. This prevents a `TypeError` and allows the app to run smoothly, simply hiding the box when no face is detected.

-----

## Future Improvements

  * **Upgrade to a Modern Face Detector:** The current implementation uses OpenCV's classic Haar Cascades. A significant improvement would be to replace this with **YuNet**, a lightweight and highly accurate deep-learning detector from the OpenCV model zoo. This would dramatically improve detection accuracy with varied angles, lighting, and partial occlusions.
  * **Support Multiple Faces:** The backend could be refactored to *always* send a 2D array (a list of faces), e.g., `[[x,y,w,h], [x2,y2,w2,h2]]`. The frontend would then use a `forEach` loop to draw a box for every face detected.

-----

## Acknowledgements & Credits

  * **AI Model:** This project uses the `emotion-ferplus-8.onnx` model from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus), which is trained on the FER+ dataset.
  * **Libraries:** This project would not be possible without the open-source work from OpenCV, Flask, ONNXRuntime, and NumPy.


```
```