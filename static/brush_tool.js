// Canvas variables
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const brushSizeInput = document.getElementById('brushSize');
const brushSizeValue = document.getElementById('brushSizeValue');
const clearButton = document.getElementById('clearButton');
const submitButton = document.getElementById('submitButton');

let isDrawing = false;
let brushSize = parseInt(brushSizeInput.value, 10);

// Load the uploaded image
const img = new Image();
img.src = uploadedImagePath;

img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
};

// Update brush size
brushSizeInput.addEventListener('input', () => {
    brushSize = parseInt(brushSizeInput.value, 10);
    brushSizeValue.textContent = brushSize;
});

// Drawing functionality
canvas.addEventListener('mousedown', () => {
    isDrawing = true;
});
canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();

    // Adjust mouse coordinates for canvas scaling
    const scaleX = canvas.width / rect.width; // Horizontal scale factor
    const scaleY = canvas.height / rect.height; // Vertical scale factor
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(x, y, brushSize, 0, Math.PI * 2);
    ctx.fill();
});

// Clear the mask
clearButton.addEventListener('click', () => {
    ctx.drawImage(img, 0, 0);
});

// Submit the image and mask
submitButton.addEventListener('click', () => {
    const maskData = canvas.toDataURL(); // Convert the canvas (mask) to an image
    console.log(maskData)
    // Send the mask and filename to the backend
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: uploadedImagePath.split('/').pop(), // Extract filename
            maskData: maskData,
        }),
    })
        .then((response) => response.blob())
        .then((blob) => {
            const resultImageUrl = URL.createObjectURL(blob);

            // Display the processed image
            const resultDiv = document.getElementById('resultDiv');
            resultDiv.innerHTML = ''; // Clear previous result
            const resultImage = document.createElement('img');
            resultImage.src = resultImageUrl;
            resultImage.classList.add('img-fluid', 'mt-3', 'shadow'); // Add Bootstrap classes
            resultDiv.appendChild(resultImage);
        })
        .catch((error) => console.error('Error:', error));
});
