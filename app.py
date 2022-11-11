from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import keras
import cv2
import numpy as np

app = Flask(__name__)
model = keras.models.load_model('potato.h5')
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def upload_image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(type(filename))
    image = cv2.imread('./static/uploads/'+filename)
    image.resize(1, 256, 256, 3)
    a=np.array(image)
    prediction = model.predict(a)
    return render_template('index.html', filename=prediction)
if __name__ == "__main__":
    app.run(debug=True)