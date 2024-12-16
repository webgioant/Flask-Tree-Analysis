from flask import Flask, request, render_template, send_file
import tensorflow as tf
import numpy as np
import rasterio
import os
from rasterio.plot import show
import matplotlib.pyplot as plt

# Load Pre-trained Model
MODEL_PATH = "tree_detection_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (OSError, IOError) as e:
    model = None
    print(f"Error loading model: {e}")

# Flask Web Server
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analisis Pohon Beringin v1.0</title>
    </head>
    <body>
        <h1>Analisis Pohon Beringin Dengan Deep Learning versi 1.0</h1>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <label for="file">Upload Citra Satelit (GeoTIFF):</label>
            <input type="file" id="file" name="file" accept=".tif" required>
            <br><br>
            <button type="submit">Analisis Citra</button>
        </form>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return "<h1>Model not available:</h1><p>The deep learning model could not be loaded. Please contact the administrator.</p><a href='/'>Back to Home</a>"

    # Save uploaded file
    uploaded_file = request.files['file']
    input_filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)

    # Check if uploaded file is a valid GeoTIFF
    try:
        with rasterio.open(uploaded_file) as src:
            pass
    except rasterio.errors.RasterioIOError:
        return "<h1>Invalid file format:</h1><p>Uploaded file is not a valid GeoTIFF.</p><a href='/'>Back to Home</a>"

    uploaded_file.save(input_filepath)

    # Process GeoTIFF
    with rasterio.open(input_filepath) as src:
        image = src.read([1, 2, 3])  # RGB bands
        profile = src.profile

    # Normalize and prepare image for prediction
    normalization_factor = 10000.0  # Ensure this matches the input data range
    image = np.transpose(image, (1, 2, 0)) / normalization_factor
    image = np.expand_dims(image, axis=0)

    # Perform prediction
    try:
        predictions = model.predict(image)
        predicted_classes = np.argmax(predictions, axis=-1).squeeze()
    except Exception as e:
        return f"<h1>Error during prediction:</h1><p>{str(e)}</p><a href='/'>Back to Home</a>"

    # Save output as GeoTIFF
    output_filepath = os.path.join(RESULT_FOLDER, f"result_{uploaded_file.filename}")
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_filepath, 'w', **profile) as dst:
        dst.write(predicted_classes.astype(rasterio.uint8), 1)

    return f'''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hasil Analisis</title>
    </head>
    <body>
        <h1>Hasil Analisis Pohon Beringin</h1>
        <p>Analisis selesai. Anda dapat mengunduh hasilnya di bawah ini:</p>
        <a href="/download/{os.path.basename(output_filepath)}" download>Unduh Hasil</a>
        <br><br>
        <a href="/">Kembali ke Halaman Utama</a>
    </body>
    </html>
    '''

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False)
