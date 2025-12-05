# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers trained model information and evaluation metrics.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import mlflow
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("register")
    parser.add_argument("--model_name", type=str, help="Name of model")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--evaluation_output", type=str, help="Path to evaluation results")
    parser.add_argument("--model_info_output_path", type=str, help="Path to save model info")

    args = parser.parse_args()
    return args


def main(args):
    '''Prepare model registration information'''

    print("=" * 80)
    print("MODEL REGISTRATION STARTED")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.model_info_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation results
    print(f"Loading evaluation results from: {args.evaluation_output}")
    eval_file = Path(args.evaluation_output) / "evaluation_results.json"
    
    eval_results = {}
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        print(f"✓ Evaluation results loaded")
        if "r2_score" in eval_results:
            print(f"  R² Score: {eval_results.get('r2_score', 'N/A'):.4f}")
        if "rmse" in eval_results:
            print(f"  RMSE: {eval_results.get('rmse', 'N/A'):.4f}")
    else:
        print("⚠ Warning: Evaluation results file not found")

    # Check if deploy flag exists
    deploy_flag = 0
    deploy_flag_file = Path(args.evaluation_output) / "deploy_flag"
    if deploy_flag_file.exists():
        with open(deploy_flag_file, 'r') as f:
            deploy_flag = int(f.read().strip())
        print(f"✓ Deploy flag loaded: {deploy_flag}")

    # Prepare model info
    print(f"\nPreparing model registration information:")
    print(f"  Model name: {args.model_name}")
    print(f"  Model path: {args.model_path}")
    
    model_info = {
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "evaluation_results": eval_results,
        "deploy_flag": deploy_flag,
        "status": "ready_for_registration" if deploy_flag == 1 else "not_ready",
        "timestamp": pd.Timestamp.now().isoformat()
    }

    # Save model registration info as JSON
    print(f"\nSaving model registration info...")
    info_file = output_dir / "model_info.json"
    with open(info_file, "w") as f:
        json.dump(model_info, f, indent=4)
    print(f"✓ Model info saved to {info_file}")

    # Log metrics to MLflow
    print(f"\nLogging metrics to MLflow...")
    mlflow.log_param("model_name", args.model_name)
    mlflow.log_param("model_path", str(args.model_path))
    
    if eval_results:
        mlflow.log_metric("registered_r2", float(eval_results.get("r2_score", 0)))
        mlflow.log_metric("registered_rmse", float(eval_results.get("rmse", 0)))
        mlflow.log_metric("registered_mae", float(eval_results.get("mae", 0)))
    
    mlflow.log_metric("deploy_flag", float(deploy_flag))
    print("✓ Metrics logged to MLflow")

    # Save model registration summary
    summary_file = output_dir / "registration_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL REGISTRATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Name: {args.model_name}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Status: {model_info['status']}\n")
        f.write(f"Deploy Flag: {deploy_flag}\n\n")
        f.write("Evaluation Metrics:\n")
        for key, value in eval_results. items():
            f.write(f"  {key}: {value}\n")
    print(f"✓ Registration summary saved to {summary_file}")

    print("\n" + "=" * 80)
    print("MODEL REGISTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    args = parse_args()

    print(f"Arguments: {args}\n")
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Evaluation output: {args.evaluation_output}")
    print(f"Model info output: {args.model_info_output_path}")
    print()

    main(args)

    mlflow.end_run()
