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
    const faceData = data.face_data;

    const container = document.querySelector(".video-wrapper");

    const oldBoxes = container.querySelectorAll(".bounding-box");
    oldBoxes.forEach(box => box.remove());


    if (faceData && faceData.length > 0){
      const naturalWidth = video.videoWidth;
      const displayedWidth = video.clientWidth;
      // calculate scale factor between natural and displayed size
      const scale = displayedWidth / naturalWidth;
      const x_original = faceData[0];
      const y_original = faceData[1];
      const w_original = faceData[2];
      const h_original = faceData[3];

      // scale the face coordinates to match displayed size
      const x = x_original * scale;
      const y = y_original * scale;
      const w = w_original * scale;
      const h = h_original * scale;

      overlay.style.left = x + 'px';
      overlay.style.top = y + 'px';
      overlay.style.display = 'block';

      const box = document.createElement("div");
      box.className = "bounding-box"; 
    
      box.style.left   = x + 'px';   
      box.style.top    = y + 'px'; 
      box.style.width  = w + 'px';   
      box.style.height = h + 'px';
      // Add the new box div inside the container
      container.appendChild(box);
      // restart the pop animation
      overlay.classList.remove('pop');
      void overlay.offsetWidth;   // force reflow
      overlay.classList.add('pop');
    } else {
      overlay.style.display = 'none';
    }

  })
  .catch(err => {
    analyzing = false;
    overlay.textContent = 'Error';
    console.error('Detection error:', err);
  });
}, 1000);
