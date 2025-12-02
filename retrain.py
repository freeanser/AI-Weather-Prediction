# retrain.py
import pymongo
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# 1. Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["weather_system"]
collection = db["prediction_logs"]

def retrain_model():
    print("--- Starting Model Iteration Process ---")
    
    # --- Step A: Load Original Training Data (Knowledge Base) ---
    print("1. Loading original dataset (weather_dataset.csv)...")
    try:
        df_old = pd.read_csv("weather_dataset.csv")
        # Ensure features match the training phase (based on your notebook)
        X_old = df_old[['Dew Point Temp_C', 'Press_kPa', 'Rel Hum_%', 'Wind Speed_km/h']]
        y_old = df_old['Temp_C']
    except FileNotFoundError:
        print("Error: 'weather_dataset.csv' not found. Please ensure the file is in the same directory.")
        return

    # --- Step B: Fetch User Data from MongoDB (Simulated Feedback Loop) ---
    print("2. Checking for new data in MongoDB...")
    data_cursor = collection.find()
    new_data_list = list(data_cursor)
    
    if len(new_data_list) > 0:
        print(f"   Found {len(new_data_list)} new user input records.")
        
        # Organize new data features
        new_features = []
        new_targets = []
        
        for doc in new_data_list:
            # Extract Features (Input)
            input_data = doc['input_features']
            features = [
                input_data['Dew_Point_Temp_C'],
                input_data['Press_kPa'],
                input_data['Rel_Hum_%'],
                input_data['Wind_Speed_km/h']
            ]
            new_features.append(features)
            
            # Since users do not provide "Actual Temperature" (Ground Truth),
            # we simulate it here to complete the retraining pipeline.
            # Assumption: Actual Temp = Predicted Temp + Random Noise (-0.5 to +0.5)
            # This allows the .fit() method to work for demonstration purposes.
            predicted_temp = doc['predicted_temperature']
            simulated_actual_temp = predicted_temp + np.random.uniform(-0.5, 0.5)
            new_targets.append(simulated_actual_temp)

        # Convert to DataFrame
        X_new = pd.DataFrame(new_features, columns=['Dew Point Temp_C', 'Press_kPa', 'Rel Hum_%', 'Wind Speed_km/h'])
        y_new = pd.Series(new_targets, name='Temp_C')
        
        print("   New data integrated with simulated ground truth labels.")

        # --- Step C: Merge Data (Data Augmentation) ---
        print("3. Merging old and new data for incremental training...")
        X_final = pd.concat([X_old, X_new], ignore_index=True)
        y_final = pd.concat([y_old, y_new], ignore_index=True)
        
    else:
        print("   No new data in MongoDB. Retraining with original data only.")
        X_final = X_old
        y_final = y_old

    # --- Step D: Retrain the Model ---
    print(f"4. Training model (Total samples: {len(X_final)})...")
    # Using parameters similar to your original notebook
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_final, y_final)
    
    # --- Step E: Save and Deploy ---
    print("5. Saving the updated model...")
    joblib.dump(model, 'trained_model.joblib')
    print(">>> Success! Model updated and saved as 'trained_model.joblib'.")
    print(">>> Please restart app.py to apply the new model.")

if __name__ == "__main__":
    retrain_model()