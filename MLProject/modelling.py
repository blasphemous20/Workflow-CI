"""
MLflow Project - Titanic Survival Prediction
Nama: Mohammad Fajar Ma'shum
Kriteria 3 - Basic: CI/CD Workflow with GitHub Actions
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score)
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" "*15 + "MLFLOW PROJECT - CI/CD WORKFLOW")
print(" "*15 + "Titanic Survival Prediction")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n📂 Step 1: Loading Data...")
print("-"*70)

try:
    X = pd.read_csv('titanic_preprocessing.csv')
    y = pd.read_csv('titanic_target.csv').values.ravel()
    print(f"✓ Data loaded successfully")
    print(f"  - Features: {X.shape}")
    print(f"  - Target: {y.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# ============================================================
# 2. SPLIT DATA
# ============================================================

print("\n🔀 Step 2: Splitting Data...")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# ============================================================
# 3. SETUP MLFLOW
# ============================================================

print("\n⚙️  Step 3: Setting up MLflow...")
print("-"*70)

# Set experiment name
experiment_name = "Titanic_CI_Workflow"
mlflow.set_experiment(experiment_name)
print(f"✓ Experiment: {experiment_name}")

# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\n🤖 Step 4: Training RandomForest Model...")
print("-"*70)

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_CI") as run:
    
    # Enable autolog
    mlflow.sklearn.autolog(log_models=True)
    
    # Define model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train model
    print("⏳ Training model...")
    model.fit(X_train, y_train)
    print("✓ Model trained successfully!")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"\n📊 Model Performance:")
    print(f"  • Accuracy:  {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    print(f"  • ROC-AUC:   {roc_auc:.4f}")
    
    # Get run info
    run_id = run.info.run_id
    artifact_uri = run.info.artifact_uri
    
    print(f"\n✓ MLflow Run ID: {run_id}")
    print(f"✓ Artifact URI: {artifact_uri}")
    
    # Save run info to file for GitHub Actions
    with open('run_info.txt', 'w') as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    
    print(f"✓ Run info saved to: run_info.txt")
    
    # Disable autolog
    mlflow.sklearn.autolog(disable=True)

# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "="*70)
print(" "*20 + "TRAINING COMPLETED")
print("="*70)

print(f"\n🎯 Model: RandomForest")
print(f"🏆 Accuracy: {accuracy:.4f}")
print(f"📁 Experiment: {experiment_name}")
print(f"🔑 Run ID: {run_id}")

print("\n✅ CI/CD WORKFLOW COMPLETED SUCCESSFULLY!")
print("="*70)
