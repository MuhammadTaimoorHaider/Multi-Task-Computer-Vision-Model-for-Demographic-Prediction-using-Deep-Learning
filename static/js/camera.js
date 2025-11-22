// Mode switching
function switchMode(mode) {
    const uploadMode = document.getElementById('uploadMode');
    const cameraMode = document.getElementById('cameraMode');
    const uploadBtn = document.getElementById('uploadModeBtn');
    const cameraBtn = document.getElementById('cameraModeBtn');
    
    if (mode === 'upload') {
        uploadMode.classList.add('active');
        cameraMode.classList.remove('active');
        uploadBtn.classList.add('active');
        cameraBtn.classList.remove('active');
        stopCamera();
    } else {
        uploadMode.classList.remove('active');
        cameraMode.classList.add('active');
        uploadBtn.classList.remove('active');
        cameraBtn.classList.add('active');
    }
    
    hideError();
}

// Upload Mode Functions
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');

uploadBox.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#e9ecef';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9fa';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9fa';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileUpload(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

function handleFileUpload(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('uploadBox').parentElement.style.display = 'none';
        document.getElementById('uploadPreview').style.display = 'block';
        
        // Send to server
        uploadImage(file);
    };
    
    reader.readAsDataURL(file);
}

function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    document.getElementById('uploadLoading').style.display = 'block';
    document.getElementById('uploadResults').style.display = 'none';
    hideError();
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('uploadLoading').style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data.predictions, 'uploadResults');
        }
    })
    .catch(error => {
        document.getElementById('uploadLoading').style.display = 'none';
        showError('Failed to analyze image. Please try again.');
        console.error('Error:', error);
    });
}

function resetUpload() {
    document.getElementById('uploadBox').parentElement.style.display = 'block';
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('uploadResults').style.display = 'none';
    fileInput.value = '';
    hideError();
}

// Camera Mode Functions
let stream = null;
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' } 
        });
        webcam.srcObject = stream;
        
        document.getElementById('startCameraBtn').style.display = 'none';
        document.getElementById('captureBtn').style.display = 'inline-block';
        document.getElementById('stopCameraBtn').style.display = 'inline-block';
        hideError();
    } catch (error) {
        showError('Camera access denied. Please allow camera permissions and try again.');
        console.error('Camera error:', error);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        stream = null;
    }
    
    document.getElementById('startCameraBtn').style.display = 'inline-block';
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('cameraResults').style.display = 'none';
}

function captureFrame() {
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0);
    
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            sendCameraFrame(reader.result);
        };
        reader.readAsDataURL(blob);
    }, 'image/jpeg');
}

function sendCameraFrame(imageData) {
    document.getElementById('cameraLoading').style.display = 'block';
    document.getElementById('cameraResults').style.display = 'none';
    hideError();
    
    fetch('/predict_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('cameraLoading').style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data.predictions, 'cameraResults');
        }
    })
    .catch(error => {
        document.getElementById('cameraLoading').style.display = 'none';
        showError('Failed to analyze frame. Please try again.');
        console.error('Error:', error);
    });
}

// Display Results
function displayResults(predictions, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    predictions.forEach((pred, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <h3>Face ${predictions.length > 1 ? index + 1 : 'Detected'}</h3>
            <div class="result-grid">
                <div class="result-item">
                    <h4>Age</h4>
                    <div class="result-value">${pred.age} years</div>
                </div>
                <div class="result-item">
                    <h4>Gender</h4>
                    <div class="result-value">${pred.gender}</div>
                    <div class="result-confidence">${pred.gender_confidence}% confidence</div>
                </div>
                <div class="result-item">
                    <h4>Ethnicity</h4>
                    <div class="result-value">${pred.ethnicity}</div>
                    <div class="result-confidence">${pred.ethnicity_confidence}% confidence</div>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
    
    container.style.display = 'block';
}

// Error handling
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}
