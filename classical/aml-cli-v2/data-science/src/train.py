# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.  
"""
Trains ML model using training dataset.  Saves trained model.   
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import gc
import warnings
warnings.filterwarnings('ignore')

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = []


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # classifier specific arguments
    parser.add_argument('--regressor__n_estimators', type=int, default=100,
                        help='Number of trees')
    parser.add_argument('--regressor__bootstrap', type=int, default=1,
                        help='Method of selecting samples for training each tree')
    parser.add_argument('--regressor__max_depth', type=int, default=15,
                        help='Maximum number of levels in tree')
    parser.add_argument('--regressor__max_features', type=str, default='sqrt',
                        help='Number of features to consider at every split')
    parser.add_argument('--regressor__min_samples_leaf', type=int, default=4,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--regressor__min_samples_split', type=int, default=5,
                        help='Minimum number of samples required to split a node')

    args = parser.parse_args()
    return args

def main(args):
    '''Read train dataset, train model, save trained model'''
    
    print("=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)
    
    # Read train data
    print(f"Loading training data from: {args.train_data}")
    train_data = pd.read_parquet(Path(args.train_data))
    print(f"Training data shape: {train_data.shape}")
    print(f"Memory usage: {train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]
    
    # Remove original dataframe to free memory
    del train_data
    gc.collect()

    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")

    # Train a Random Forest Regression Model with the training set
    print("\nInitializing Random Forest Regressor...")
    print(f"  n_estimators: {args.regressor__n_estimators}")
    print(f"  max_depth: {args.regressor__max_depth}")
    print(f"  max_features: {args.regressor__max_features}")
    
    model = RandomForestRegressor(
        n_estimators=args.regressor__n_estimators,
        bootstrap=args.regressor__bootstrap,
        max_depth=args.regressor__max_depth,
        max_features=args.regressor__max_features,
        min_samples_leaf=args.regressor__min_samples_leaf,
        min_samples_split=args.regressor__min_samples_split,
        random_state=0,
        n_jobs=-1,
        verbose=1
    )

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Model training completed")

    # Predict using the Regression Model
    print("Generating predictions...")
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    print("Evaluating model...")
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    print("Creating visualization...")
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, yhat_train, color='black', alpha=0.5)
        plt.plot(y_train, y_train, color='blue', linewidth=3)
        plt.xlabel("Real value")
        plt.ylabel("Predicted value")
        plt.title("Model Predictions vs Actual Values")
        plt.tight_layout()
        plt.savefig("regression_results.png", dpi=100)
        print("✓ Visualization saved")
    except Exception as e:
        print(f"⚠ Warning: Could not log artifact: {str(e)}")

    # Save the model
    print(f"Saving model to: {args.model_output}")
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)
    print("✓ Model saved successfully")
    
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
