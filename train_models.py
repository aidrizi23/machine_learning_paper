"""
Model Training and Evaluation Module
Trains SVM, Decision Tree, Random Forest, and Linear Regression models
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import pickle
import os
from preprocess import load_and_preprocess_data

def train_and_evaluate_models():
    """
    Train all models and evaluate their performance
    """
    # Load preprocessed data
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    # Initialize models
    models = {
        'Support Vector Machine (SVM)': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
    }

    results = []

    # Train and evaluate each model
    for model_name, model in models.items():
        print("\n" + "="*70)
        print(f"TRAINING: {model_name}")
        print("="*70)

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time (s)': train_time
        })

        print(f"Training Time: {train_time:.4f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Save model
        os.makedirs('models', exist_ok=True)
        model_filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_filename}")

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\nResults saved to results/model_comparison.csv")

    return results_df

if __name__ == "__main__":
    results = train_and_evaluate_models()
