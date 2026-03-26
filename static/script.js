const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const socket = io();

let currentEmotion = '';
let currentBox = null;
let lastEmit = 0;
const EMIT_INTERVAL = 200;

socket.on('emotion', ({ emotion }) => {
  currentEmotion = emotion;
});

// Draw loop — runs at full framerate, independent of detection speed
function drawLoop() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (currentBox) {
    const { x, y, width, height } = currentBox;

    // Bounding box
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);

    // Emotion label anchored to bounding box
    if (currentEmotion) {
      const label = currentEmotion.charAt(0).toUpperCase() + currentEmotion.slice(1);
      ctx.font = 'bold 15px Arial, sans-serif';
      const textW = ctx.measureText(label).width + 12;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
      ctx.fillRect(x, y - 26, textW, 24);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 6, y - 8);
    }
  }

  requestAnimationFrame(drawLoop);
}

// Detection loop — runs as fast as face-api can process
async function detectionLoop() {
  const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });

  while (true) {
    const detections = await faceapi.detectAllFaces(video, options);

    if (detections.length > 0) {
      currentBox = detections[0].box;

      const now = Date.now();
      if (now - lastEmit >= EMIT_INTERVAL) {
        lastEmit = now;
        emitFace(currentBox);
      }
    } else {
      currentBox = null;
    }
  }
}

function emitFace(box) {
  const tmp = document.createElement('canvas');
  tmp.width = 64;
  tmp.height = 64;
  tmp.getContext('2d').drawImage(
    video,
    box.x, box.y, box.width, box.height,
    0, 0, 64, 64
  );
  const base64 = tmp.toDataURL('image/jpeg', 0.8).split(',')[1];
  socket.emit('frame', base64);
}

async function init() {
  await faceapi.nets.tinyFaceDetector.loadFromUri(
    'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model'
  );

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  await new Promise(resolve => video.addEventListener('loadedmetadata', resolve, { once: true }));

  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  requestAnimationFrame(drawLoop);
  detectionLoop();
}

init().catch(err => console.error('Init failed:', err));
