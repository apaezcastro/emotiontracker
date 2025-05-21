const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const emotionText = document.getElementById('emotionText');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

// Capture and send frame every 2 seconds
setInterval(() => {
    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg');
    const startTime = performance.now();
    fetch('/detect_emotion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        const duration = (performance.now() - startTime).toFixed(2);
    console.log(`Emotion detected: ${data.emotion} (took ${duration}ms)`);
    if ('confidence' in data) {
        console.log(`Confidence: ${(data.confidence * 100).toFixed(2)}%`);
    }
    if ('raw_probabilities' in data) {
        console.log('All probabilities:', data.raw_probabilities);
    }
    emotionText.textContent = `Current Emotion: ${data.emotion}`;
});
}, 2000);
