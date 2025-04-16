from io import BytesIO  
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# Load models and data
species_model = tf.keras.models.load_model("species_classifier_model1.keras")
flammability_model = joblib.load("flammability_predictor.pkl")
scaler = joblib.load("scaler.pkl")

with open("species_class_map.json") as f:
    index_to_species = json.load(f)
with open("flammable_species_list.json") as f:
    valid_species = json.load(f)

def preprocess_image(img_file):
    img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def categorize_risk(score):
    if score >= 20:
        return "High"
    elif score >= 10:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img_data = preprocess_image(img_file)

    pred_probs = species_model.predict(img_data)[0]
    pred_index = np.argmax(pred_probs)
    raw_species_name = index_to_species[str(pred_index)]

    if raw_species_name.lower() not in [s.lower() for s in valid_species]:
        return jsonify({
            'species': raw_species_name,
            'flammability_info': 'Not available for this species'
        })

    dummy_input = np.zeros((1, scaler.mean_.shape[0]))
    dummy_input = scaler.transform(dummy_input)
    flammability_score = flammability_model.predict(dummy_input)[0]
    flammability_category = categorize_risk(flammability_score)

    return jsonify({
        'species': raw_species_name,
        'flammability_score': round(float(flammability_score), 2),
        'flammability_level': flammability_category
    })

if __name__ == '__main__':
    app.run(debug=True)
