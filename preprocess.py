"""
Data Preprocessing Module
Loads and prepares the wine dataset for machine learning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(filepath='data/winequality-red.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess the wine quality dataset

    Args:
        filepath: Path to the dataset
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed train and test sets
        feature_names: List of feature names
    """
    # Load dataset
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")

    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Separate features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    feature_names = X.columns.tolist()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    print("\nPreprocessing completed successfully!")
