# train_model.py
# Script to train the heart disease prediction model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from pathlib import Path

def preprocess_data(df):
    """Preprocess the data with feature engineering"""
    # Feature Engineering
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    if 'BP' in df.columns:
        df['BP_Category'] = pd.cut(df['BP'], bins=[0, 120, 140, 300], labels=[0, 1, 2]).astype(int)
    if 'Cholesterol' in df.columns:
        df['Chol_Risk'] = (df['Cholesterol'] > 200).astype(int)
    if 'Max HR' in df.columns:
        df['HR_Risk'] = (df['Max HR'] < 100).astype(int)

    return df

def train_model():
    """Train the heart disease prediction model"""
    try:
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        # For demonstration, create a synthetic dataset
        # In real scenario, load from CSV: df = pd.read_csv('heart.csv')
        np.random.seed(42)
        n_samples = 1000

        # Generate synthetic data similar to heart disease dataset
        # Using the same column names as expected by the API
        data = {
            'Age': np.random.normal(55, 10, n_samples).clip(20, 80),
            'Sex': np.random.choice([0, 1], n_samples),
            'Chest pain type': np.random.choice([1, 2, 3, 4], n_samples),  # 1-4 as in API
            'BP': np.random.normal(130, 20, n_samples).clip(90, 200),
            'Cholesterol': np.random.normal(240, 50, n_samples).clip(100, 400),
            'FBS over 120': np.random.choice([0, 1], n_samples),
            'EKG results': np.random.choice([0, 1, 2], n_samples),
            'Max HR': np.random.normal(150, 25, n_samples).clip(70, 200),
            'Exercise angina': np.random.choice([0, 1], n_samples),
            'ST depression': np.random.exponential(1, n_samples).clip(0, 6),
            'Slope of ST': np.random.choice([1, 2, 3], n_samples),  # 1-3 as in API
            'Number of vessels fluro': np.random.choice([0, 1, 2, 3], n_samples),
            'Thallium': np.random.choice([3, 6, 7], n_samples),
        }

        df = pd.DataFrame(data)

        # Create target variable (simplified logic)
        risk_factors = (
            (df['Age'] > 60).astype(int) +
            (df['BP'] > 140).astype(int) +
            (df['Cholesterol'] > 240).astype(int) +
            (df['FBS over 120'] == 1).astype(int) +
            (df['Max HR'] < 120).astype(int) +
            df['Exercise angina'] +
            (df['ST depression'] > 2).astype(int)
        )
        df['target'] = (risk_factors > 2).astype(int)

        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution: {df['target'].value_counts()}")

        # Preprocess data
        df_processed = preprocess_data(df.copy())

        # Define features (original + engineered)
        feature_cols = [
            'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results',
            'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium',
            'Age_Group', 'BP_Category', 'Chol_Risk', 'HR_Risk'
        ]

        X = df_processed[feature_cols]
        y = df_processed['target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model and artifacts
        joblib.dump(model, models_dir / 'heart_disease_model.pkl')
        joblib.dump(scaler, models_dir / 'scaler.pkl')
        joblib.dump(feature_cols, models_dir / 'feature_names.pkl')

        print("Model and artifacts saved successfully!")
        print(f"Features used: {len(feature_cols)}")
        print(f"Training completed with {accuracy:.1%} accuracy on test set")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()