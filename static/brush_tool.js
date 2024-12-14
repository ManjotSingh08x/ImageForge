const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('image');
const canvas = document.getElementById('canvas');
const editor = document.getElementById('editor');
const errorDiv = document.getElementById('error');
const brushSizeInput = document.getElementById('brushSize');
const brushSizeValue = document.getElementById('brushSizeValue');
const clearButton= document.getElementById('clearButton');
const submitButton = document.getElementById('submitButton');

// Canvas variables
const ctx = canvas.getContext('2d');
let isDrawing = false;
let brushSize = parseInt(brushSizeInput.value, 10);

brushSizeInput.addEventListener('input', () => {
    brushSize = parseInt(brushSizeInput.value, 10);
    brushSizeValue.textContent = brushSize;
});

document.getElementById('uploadButton').addEventListener('click', ()=> {
    const file = imageInput.files[0];
    if (!file) {
        errorDiv.textContent = "Please select a file."
        return;
    }
    if (file.type !== 'image/png'){
        errorDiv.textContent = 'The file msut be PNG image.';
        return;
    }

    const img = new Image();
    img.src = URLcreateObjectURL(file);
    img.onload = () => {
        if (img.width !== 256 || img.height !== 256) {
            errorDiv.textContent = "The Image must be exactly 256x256 pixels";
            return
        }

        errorDiv.textContent = '';
        editor.style.display = "block";

        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };

    img.onerror = () => {
        errorDiv. textContent = "Invalid image file.";
    };
});

canvas.addEventListener('mousedown', () => {
    isDrawing = true;
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.beginPath();
});

canvas.addEventListener('mousemove',(e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(x, y, brushsize, 0, Math.pi * 2);
    ctx.fill();
});

clearButton.addEventListener('click', () => {
    const img = new Image();
    img.src = URL.createObjectURL(imageInput.files[0]);
    img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
    };
})

submitButton.addEventListener('click', () => {
    canvas.toBlob((maskblob) => {
        const formData = new FormData();
        formData.append('mask', maskblob);
        formData.append('image', imageInput.files[0]);

        fetch('/inpaint', {
            method: 'POST',
            body: formData
        })
        .then((response) => response.json())
        .then((data) => {
            alert("Inpainting successful!");
            console.log(data);
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Inpainting failed!");
        });
    });
});
