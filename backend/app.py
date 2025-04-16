from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Load your trained model
model = joblib.load(r"C:\Users\admin\OneDrive\Desktop\dsbda_mini\model\uber_model.pkl")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        features = np.array([[float(data['pickup_longitude']),
                              float(data['pickup_latitude']),
                              float(data['dropoff_longitude']),
                              float(data['dropoff_latitude']),
                              int(data['passenger_count'])]])

        fare = model.predict(features)[0]
        return jsonify({'fare': round(float(fare), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
