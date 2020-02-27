
const image = document.getElementById('image'); 
const canvas = document.getElementById('canvas');
const dropContainer = document.getElementById('container');
const warning = document.getElementById('warning');
const fileInput = document.getElementById('fileUploader');

const id2class = {0:"有口罩", 1:"无口罩"};
let model;

function preventDefaults(e) {
  e.preventDefault()
  e.stopPropagation()
};

function windowResized() {
  let windowW = window.innerWidth;
  if (windowW < 480 && windowW >= 200) {
    dropContainer.style.display = 'block';
  } else if (windowW < 200) {
    dropContainer.style.display = 'none';
  } else {
    dropContainer.style.display = 'block';
  }
}

['dragenter', 'dragover'].forEach(eventName => {
  dropContainer.addEventListener(eventName, e => dropContainer.classList.add('highlight'), false)
});

['dragleave', 'drop'].forEach(eventName => {
  dropContainer.addEventListener(eventName, e => dropContainer.classList.remove('highlight'), false)
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropContainer.addEventListener(eventName, preventDefaults, false)
});

dropContainer.addEventListener('drop', gotImage, false)


function gotImage(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 1) {
    console.error('upload only one file');
  }
  const file = files[0];
  const imageType = /image.*/;
  if (file.type.match(imageType)) {
    warning.innerHTML = '';
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
      image.src = reader.result;
      setTimeout(detectImage, 100);
    }
  } else {
    image.src = 'images/demo.jpg';
    setTimeout(detectImage, 100);
    warning.innerHTML = 'Please drop an image file.'
  }
}

function handleFiles() {
  const curFiles = fileInput.files;
  if (curFiles.length === 0) {
    image.src = 'images/demo.jpg';
    setTimeout(detectImage, 100);
    warning.innerHTML = 'No image selected for upload';
  } else {
    image.src = window.URL.createObjectURL(curFiles[0]);
    warning.innerHTML = '';
    setTimeout(detectImage, 100);
  }
}

function clickUploader() {
  fileInput.click();
}

// 检测人脸和口罩
function detectImage() {
  detect(image).then((results) => {
    canvas.width = image.width;
    canvas.height = image.height;
    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    for(bboxInfo of results) {
      bbox = bboxInfo[0];
      classID = bboxInfo[1];
      score = bboxInfo[2];

      ctx.beginPath();
      ctx.lineWidth="4";
      if (classID == 0) {
          ctx.strokeStyle="green";
          ctx.fillStyle="green";
      } else {
          ctx.strokeStyle="red";
          ctx.fillStyle="red";
      }
      
      ctx.rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
      ctx.stroke();
      
      ctx.font="30px Arial";
      
      let content = id2class[classID] + " " + score.toFixed(2);
      ctx.fillText(content, bbox[0], bbox[1] < 20 ? bbox[1] + 30 : bbox[1]-5);
  }
  })
}

// 初始化函数
async function setup() {
  await loadModel();
  // Make a detection with the default image
  detectImage();
}

setup();
