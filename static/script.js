const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlay = document.getElementById('overlay');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(err => {
    overlay.textContent = 'Camera access denied';
    console.error(err);
  });

let analyzing = false;
setInterval(() => {
  if (analyzing || video.videoWidth === 0) return;
  analyzing = true;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  const imageData = canvas.toDataURL('image/jpeg');
  fetch('/detect_emotion', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  })
  .then(res => {
    analyzing = false;
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    return res.json();
  })
  .then(data => {
    overlay.textContent = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);

    // restart the pop animation
    overlay.classList.remove('pop');
    void overlay.offsetWidth;   // force reflow
    overlay.classList.add('pop');

  })
  .catch(err => {
    analyzing = false;
    overlay.textContent = 'Error';
    console.error('Detection error:', err);
  });
}, 1000);
