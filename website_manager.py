from flask import Flask, render_template, request, send_file, redirect, url_for
from PIL import Image
from PartialConvArch import *
import pickle
import torch
import os

app = Flask(__name__)

OUTPUT_FOLDER = 'static/output'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if not file.filename.lower().endswith('.jpg'):
        return "File must be a JPEG image.", 400
    
    try:
        image = Image.open(file)
        if image.size != (256, 256):
            return "Image must be 256x256 pixels", 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(filepath)
        return redirect(url_for('editing', filename=file.filename))
    except Exception as e:
        return f"An error occured: {e}", 400
    
@app.route('/editing/<filename>')
def editing(filename):
    return render_template('editing.html', filename=filename)
   
@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()

    filename = data.get('filename')
    mask_data = data.get('maskData')


    #the code to do the processing of the image will come here 
    #for now this will only convert the image to grayscale and return the result, 
    # but after testing the ai model we will process it and return the inpainted result
    model = PartialConvUNet()
    model= torch.load('models/256resoModel_dict', map_location=torch.device('cpu'))

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    img = Image.open(file_path)
    img, mask = getinput(img, mask_data)
    img = model(img.to(dtype=float), mask)
    output_path = os.path.join(OUTPUT_FOLDER, "@" + filename)
    img.save(output_path)

    return send_file(output_path, mimetype='image/jpg')



if __name__ == '__main__':
    app.run(debug=True, port=5001)