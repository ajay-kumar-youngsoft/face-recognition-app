Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  document.body.append("Models are loaded!");
  let video
  
  // Get access to the camera
  try {
    video = await navigator.mediaDevices.getUserMedia({ video: {} })
  } catch (e) {
    console.error('Failed to access the webcam', e)
    return
  }
  
  // Create a video element and start playing the stream
  const videoEl = document.createElement('video')
  videoEl.srcObject = video
  videoEl.play()
  
  // Perform face recognition on each frame of the video
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(videoEl).withFaceLandmarks().withFaceDescriptors()
    const results = detections.map(d => faceMatcher.findBestMatch(d.descriptor))
    results.forEach((result, i) => {
      console.log(`Person: ${result.label}, Confidence: ${result.distance}`)
    })
  }, 500)
}

async function loadLabeledImages() {
  const labels = ['Amaresh', 'Ajay Kumar', 'Madhumita', 'Siva Sai'];
  const descriptions = [];

  for (const label of labels) {
    const labelDescriptions = [];

    for (let i = 1; i <= 5; i++) {
      const img = await loadImageFromFile(`./labeled_images/${label}/${i}.jpg`);
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
      labelDescriptions.push(detections.descriptor);
    }

    descriptions.push(new faceapi.LabeledFaceDescriptors(label, labelDescriptions));
  }

  return descriptions;
}

async function loadImageFromFile(path) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = path;
  });
}