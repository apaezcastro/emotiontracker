const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const emotionText = document.getElementById('emotionText');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

// Capture and send frame every 2 seconds
setInterval(() => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg');

    fetch('/detect_emotion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        emotionText.textContent = `Current Emotion: ${data.emotion}`;
    });
}, 2000);
