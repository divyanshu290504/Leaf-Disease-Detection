from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import keras
import numpy as np
from PIL import Image
import tensorflow as tf

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
    img = Image.open('./static/uploads/'+filename)
    image = np.asarray(img)
    a= tf.convert_to_tensor(image)
    image=a.numpy().astype("uint8")
    image.resize(1, 256, 256, 3)
    b=np.array(image)
    potato_class=['Early Blight','Healthy','Late Blight']
    #prediction = potato_class[np.argmax(model.predict(a)[0])]
    prediction = model.predict(b)
    return render_template('index.html', filename=prediction)
if __name__ == "__main__":
    app.run(debug=True)
