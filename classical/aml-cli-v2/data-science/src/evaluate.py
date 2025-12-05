# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.  
"""
Evaluates trained ML model using test dataset.  
Saves predictions, evaluation results and deploy flag.  
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
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

CAT_ORD_COLS = [
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")
    parser.add_argument("--runner", type=str, help="Local or Cloud Runner", default="CloudRunner")

    args = parser.parse_args()

    return args

def main(args):
    '''Read trained model and test dataset, evaluate model and save result'''

    print("=" * 80)
    print("EVALUATION STARTED")
    print("=" * 80)

    # Load the test data
    print(f"Loading test data from: {args.test_data}")
    test_data = pd.read_parquet(Path(args.test_data))
    print(f"Test data shape: {test_data.shape}")

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Load the model from input port
    print(f"Loading model from: {args.model_input}")
    model = mlflow.sklearn.load_model(args.model_input)
    print("✓ Model loaded successfully")

    # Create output directory
    output_dir = Path(args.evaluation_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Model Evaluation ---------------- #
    yhat_test, score = model_evaluation(X_test, y_test, model, args.evaluation_output)

    # ----------------- Model Promotion ---------------- #
    if args.runner == "CloudRunner":
        print("\nEvaluating model promotion...")
        predictions, deploy_flag = model_promotion(args.model_name, args.evaluation_output, X_test, y_test, yhat_test, score)

    print("=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


def model_evaluation(X_test, y_test, model, evaluation_output):
    '''Evaluate model on test data'''

    print("\nEvaluating model on test data...")

    # Get predictions to y_test (y_test)
    yhat_test = model.predict(X_test)

    # Create output directory
    output_dir = Path(evaluation_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the output data with feature columns, predicted cost, and actual cost in csv file
    print("Saving predictions...")
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((output_dir / "predictions.csv"))
    print(f"✓ Predictions saved to {output_dir / 'predictions.csv'}")

    # Evaluate Model performance with the test set
    r2 = r2_score(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, yhat_test)

    print(f"\nEvaluation Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # Print score report to a text file
    print("Saving score report...")
    (output_dir / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}\n"
    )
    with open((output_dir / "score.txt"), "a") as outfile:
        outfile.write(f"Mean squared error: {mse:.2f}\n")
        outfile.write(f"Root mean squared error: {rmse:.2f}\n")
        outfile.write(f"Mean absolute error: {mae:.2f}\n")
        outfile.write(f"Coefficient of determination: {r2:.2f}\n")

    # Log metrics to MLflow
    mlflow.log_metric("test r2", r2)
    mlflow.log_metric("test mse", mse)
    mlflow.log_metric("test rmse", rmse)
    mlflow.log_metric("test mae", mae)

    # Visualize results (save locally, don't log to MLflow)
    print("Creating visualization...")
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, yhat_test, color='black')
        plt.plot(y_test, y_test, color='blue', linewidth=3)
        plt.xlabel("Real value")
        plt.ylabel("Predicted value")
        plt.title("Comparing Model Predictions to Real values - Test Data")
        plt.savefig(output_dir / "predictions.png", dpi=100)
        plt.close()
        print(f"✓ Visualization saved to {output_dir / 'predictions.png'}")
    except Exception as e:
        print(f"⚠ Warning: Could not save visualization: {str(e)}")

    return yhat_test, r2

def model_promotion(model_name, evaluation_output, X_test, y_test, yhat_test, score):
    '''Compare current model with previous versions for promotion decision'''
    
    print(f"Checking for previous model versions of '{model_name}'...")
    
    scores = {}
    predictions = {}

    try:
        client = MlflowClient()

        for model_run in client.search_model_versions(f"name='{model_name}'"):
            model_version = model_run.version
            print(f"  Found version {model_version}")
            try:
                mdl = mlflow.pyfunc.load_model(
                    model_uri=f"models:/{model_name}/{model_version}")
                predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
                scores[f"{model_name}:{model_version}"] = r2_score(
                    y_test, predictions[f"{model_name}:{model_version}"])
            except Exception as e:
                print(f"  ⚠ Could not load version {model_version}: {str(e)}")
                continue

        if scores:
            print(f"\nComparing with {len(scores)} previous version(s)...")
            if score >= max(list(scores.values())):
                deploy_flag = 1
                print("✓ Current model is better - Ready for deployment")
            else:
                deploy_flag = 0
                print("✗ Previous model is better - Not ready for deployment")
        else:
            deploy_flag = 1
            print("✓ No previous versions found - Ready for deployment")
    except Exception as e:
        print(f"⚠ Warning: Could not check previous versions: {str(e)}")
        deploy_flag = 1

    print(f"\nDeploy flag: {deploy_flag}")

    # Save deploy flag
    output_dir = Path(evaluation_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open((output_dir / "deploy_flag"), 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")
    print(f"✓ Deploy flag saved")

    # add current model score and predictions
    scores["current model"] = score
    predictions["current model"] = yhat_test

    # Create performance comparison plot (save locally, don't log to MLflow)
    try:
        perf_comparison_plot = pd.DataFrame(
            scores, index=["r2 score"]).plot(kind='bar', figsize=(15, 10))
        perf_comparison_plot.figure.savefig(output_dir / "perf_comparison.png", dpi=100)
        plt.close()
        print("✓ Performance comparison plot saved")
    except Exception as e:
        print(f"⚠ Warning: Could not save comparison plot: {str(e)}")

    # Log deploy flag metric (but NOT the artifact)
    mlflow.log_metric("deploy flag", bool(deploy_flag))

    return predictions, deploy_flag

if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_input}",
        f"Test data path: {args.test_data}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)
    
    print()
    main(args)

    mlflow.end_run()
