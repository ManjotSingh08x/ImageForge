from flask import Flask, render_template, request
from PIL import Image
import os

app = Flask(__name__)

OUTPUT_FOLDER = 'static/output'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if not file.filename.lower().endswith('.png'):
        return "File must be a PNG image.", 400
    
    try:
        image = Image.open(file)
        if image.size != (256, 256):
            return "Image must be 256x256 pixels", 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(filepath)
        return f"Image uploaded successfully: <a href='/{filepath}'>View Image</a>"
    except Exception as e:
        return f"An error occured: {e}", 400
    
@app.route('/inpaint', methods=['POST'])
def inpaint():
    if 'image' not in request.files:
        return {"error": "No image file uploaded"}, 400
    
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return {'error': "No selected file"}, 400
    
    # Save the uploaded image temporarily
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    image = Image.open(file_path)
    grayscale_image = image.convert('L')

    output_path = os.path.join(OUTPUT_FOLDER, uploaded_file.filename)
    grayscale_image.save(output_path)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5001)