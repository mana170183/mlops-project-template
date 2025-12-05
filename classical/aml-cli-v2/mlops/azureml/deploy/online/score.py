import os
import json
import numpy as np
import joblib

def init():
    global model
    try:
        # Get the model directory (which contains taxi-model subdirectory)
        model_base_dir = os.getenv("AZUREML_MODEL_DIR")
        
        # List what's in the model directory
        print(f"Model base directory: {model_base_dir}")
        if os.path.exists(model_base_dir):
            print(f"Contents: {os.listdir(model_base_dir)}")
        
        # Try loading from the taxi-model subdirectory
        model_dir = os.path.join(model_base_dir, "taxi-model")
        model_path = os.path.join(model_dir, "model.pkl")
        
        print(f"Loading model from: {model_path}")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✓ Model loaded successfully")
        else:
            print(f"ERROR: Model file not found at {model_path}")
            print(f"Directory contents: {os.listdir(model_dir) if os.path.exists(model_dir) else 'DIR NOT FOUND'}")
            raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def run(raw_data):
    try:
        data = json.loads(raw_data)
        
        features = np.array([[
            float(data.get("distance", 0)),
            float(data.get("dropoff_latitude", 0)),
            float(data.get("dropoff_longitude", 0)),
            float(data.get("passengers", 1)),
            float(data.get("pickup_latitude", 0)),
            float(data.get("pickup_longitude", 0)),
            float(data.get("pickup_weekday", 0)),
            float(data.get("pickup_month", 0)),
            float(data.get("pickup_monthday", 0)),
            float(data.get("pickup_hour", 0)),
            float(data.get("pickup_minute", 0)),
            float(data.get("pickup_second", 0)),
            float(data.get("dropoff_weekday", 0)),
            float(data.get("dropoff_month", 0)),
            float(data.get("dropoff_monthday", 0)),
            float(data.get("dropoff_hour", 0)),
            float(data.get("dropoff_minute", 0)),
            float(data.get("dropoff_second", 0)),
            1.0 if data.get("store_forward", False) else 0.0,
            1.0 if data.get("vendor", "CMT") == "CMT" else 0.0
        ]])
        
        prediction = model.predict(features)[0]
        
        return json.dumps({
            "prediction": float(prediction),
            "predicted_fare": round(float(prediction), 2)
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return json.dumps({"error": str(e)})
