# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pymongo
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.joblib')

MONGO_URI = "mongodb://localhost:27017/" 
client = pymongo.MongoClient(MONGO_URI)

db = client["weather_system"]
feedback_collection = db["prediction_logs"]

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # The method of extracting JSON data from an HTTP request
        data = request.get_json()
        
        # Extract features exactly as before
        features = [
            data['Dew_Point_Temp_C'],
            data['Press_kPa'],
            data['Rel_Hum_%'],
            data['Wind_Speed_km/h']
        ]

        # Converts the list of features into a NumPy array and reshapes it into a 2D array with one row and as many columns as there are features.
        data_array = np.array(features).reshape(1, -1)

        # Make predictions using the model
        prediction = model.predict(data_array)
        predicted_value = float(prediction[0]) # converts into Python float to store in MongoDB

        # Feedback Loop Implementation: persisting user inputs"
        log_entry = {
            "input_features": data,
            "predicted_temperature": predicted_value,
            "timestamp": datetime.utcnow(), 
            "model_version": "v1.0"
        }
        
        # MongoDB
        feedback_collection.insert_one(log_entry)
        print(f"Logged prediction to MongoDB: {predicted_value}")

        # Convert the prediction result to JSON format and return it
        return jsonify({'prediction': [predicted_value]})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)


#     # The method of extracting JSON data from an HTTP request. In this code, it is used to retrieve JSON data from a POST request sent by the user, which contains the feature values used for prediction.
#     data = request.get_json()
#     # Extract features and convert to NumPy array
#     features = [
#         data['Dew_Point_Temp_C'],
#         data['Press_kPa'],
#         data['Rel_Hum_%'],
#         data['Wind_Speed_km/h']
#     ]

#     # Converts the list of features into a NumPy array and reshapes it into a 2D array with one row and as many columns as there are features.
#     data_array = np.array(features).reshape(1, -1)

#     # Make predictions using the model
#     prediction = model.predict(data_array)

#     # Convert the prediction result to JSON format and return it
#     return jsonify({'prediction': prediction.tolist()})

# if __name__ == '__main__':
#     # Run the Flask app in debug mode
#     app.run(debug=True)
