# Machine Learning Weather Prediction System

An end-to-end Machine Learning web application designed to predict real-time temperature based on meteorological data. This project demonstrates a complete MLOps pipeline, integrating a **Flask** backend, **MongoDB** for data logging, and an automated retraining mechanism to simulate a continuous learning feedback loop.

## 🚀 Features

* **Real-time Prediction**: Web interface to input weather parameters (Dew Point, Pressure, Humidity, Wind Speed) and get instant temperature predictions.
* **Machine Learning Model**: Built with **Scikit-learn** using **Random Forest Regressor** (optimized via Hyperparameter Tuning) and **Lasso Regression**, achieving high accuracy with a significant reduction in RMSE.
* **Data Persistence**: Integrates **MongoDB** to log all user requests and prediction results for monitoring and future analysis.
* **MLOps & Continuous Learning**: Includes a `retrain.py` script that implements a feedback loop. It fetches new data from MongoDB, simulates ground truth labels, merges with historical data, and retrains the model automatically.
* **RESTful API**: The backend is structured to serve predictions via API endpoints.

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Web Framework**: Flask
* **Database**: MongoDB (NoSQL)
* **Machine Learning**: Scikit-learn, Pandas, NumPy, Joblib
* **Visualization**: Matplotlib, Seaborn (used in analysis notebooks)


## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/weather-prediction.git](https://github.com/your-username/weather-prediction.git)
cd weather-prediction
````

### 2. Set up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure MongoDB

Ensure you have MongoDB installed and running locally.

```bash
# Start MongoDB service (command varies by OS)
mongod --dbpath /data/db
```

*The application connects to `mongodb://localhost:27017/` by default.*

## Usage

### 1. Run the Web Application

Start the Flask server:

```bash
python app.py
```

  * Access the application at `http://127.0.0.1:5000/`.
  * Enter weather values (e.g., Dew Point, Pressure, Humidity) and click **Predict**.
  * The prediction is displayed, and the input data is automatically logged to MongoDB.

### 2. Simulate Model Retraining (MLOps Pipeline)

To demonstrate the feedback loop and model update process:

1.  Generate some traffic by making predictions on the web app.
2.  Run the retraining script:
    ```bash
    python retrain.py
    ```
3.  The script will:
      * Fetch new data from MongoDB.
      * Simulate "Ground Truth" values (since actual real-time weather data is unavailable in this demo).
      * Merge new data with the original `weather_dataset.csv`.
      * Retrain the Random Forest model.
      * Update `trained_model.joblib`.
4.  Restart `app.py` to serve the updated model.

## Model Performance

  * **Algorithm**: Random Forest Regressor
  * **Optimization**: Hyperparameter tuning was performed to optimize `n_estimators` and `max_depth`.
  * **Metric**: The model achieved a **20% improvement in Validation RMSE** compared to the baseline Linear Regression model.

## License

This project is open-source and available under the MIT License.
