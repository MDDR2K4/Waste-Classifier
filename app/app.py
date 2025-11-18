import os
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'waste_classifier.keras')
model = tf.keras.models.load_model(model_path)

# Define the class names (ensure this order is the same as during training)
CLASS_NAMES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'app/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(img_path, model):
    """
    Preprocesses the image and makes a prediction.
    """
    img = Image.open(img_path).resize((224, 224))
    
    # Convert image to numpy array
    x = np.array(img)
    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image for the model (optional, if you used a preprocess_input function)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class_index = np.argmax(preds)
        result = CLASS_NAMES[pred_class_index]
        
        # Pass the prediction and image path to the result page
        return render_template('index.html', prediction=result, image_path=os.path.join('static', 'uploads', secure_filename(f.filename)))
    return None

if __name__ == '__main__':
    app.run(debug=True)