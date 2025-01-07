document.getElementById('uploadForm').addEventListener('submit', function(event){
    const fileInput = document.getElementById('image');
    const errorDiv = document.getElementById('error');
    const file = fileInput.files[0];

    if (!file) {
        errorDiv.textContent = 'Please select a file.';
        event.preventDefault();
        return;
    }

    if (file.type != "image/jpeg"){
        errorDiv.textContent = "The file must be a JPEG image.";
        event.preventDefault();
        return;
    }
    const image = new Image();
    image.src = URL.createObjectURL(file);
    image.onload = function(){
        if (image.width !== 256 || image.height !== 256){
            errorDiv.textContent = "The image must be 256x256 pixels";
            event.preventDefault();
        } else{
            document.getElementById('uploadForm').submit();
        }
    };
    event.preventDefault()
});