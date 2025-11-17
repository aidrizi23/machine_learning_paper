"""
Wine Quality Prediction - ML Model Training and Comparison
This script trains and compares multiple machine learning models for wine quality prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_explore_data(filepath):
    """Load and perform exploratory data analysis"""
    print("="*80)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("="*80)

    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {df.shape[1]}")
    print(f"  Samples: {df.shape[0]}")

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- First Few Rows ---")
    print(df.head())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Quality Distribution ---")
    print(df['quality'].value_counts().sort_index())

    # Simplify quality to binary classification (good vs bad wine)
    # Quality <= 5 = 0 (bad), Quality > 5 = 1 (good)
    df['quality_binary'] = (df['quality'] > 5).astype(int)
    print("\n--- Binary Quality Distribution ---")
    print(df['quality_binary'].value_counts())

    return df

def visualize_data(df):
    """Create visualizations for exploratory data analysis"""
    print("\n" + "="*80)
    print("STEP 2: DATA VISUALIZATION")
    print("="*80)

    # Quality distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original quality distribution
    df['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Wine Quality Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')

    # Binary quality distribution
    df['quality_binary'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['salmon', 'lightgreen'])
    axes[0, 1].set_title('Binary Quality Distribution (0=Poor, 1=Good)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Quality')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticklabels(['Poor (≤5)', 'Good (>5)'], rotation=0)

    # Wine type distribution
    df['wine_type'].value_counts().plot(kind='bar', ax=axes[1, 0], color=['darkred', 'wheat'])
    axes[1, 0].set_title('Wine Type Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Wine Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticklabels(['Red', 'White'], rotation=0)

    # Alcohol vs Quality
    quality_groups = df.groupby('quality')['alcohol'].mean().sort_index()
    quality_groups.plot(kind='line', marker='o', ax=axes[1, 1], color='purple', linewidth=2)
    axes[1, 1].set_title('Average Alcohol Content by Quality', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Quality Score')
    axes[1, 1].set_ylabel('Average Alcohol %')

    plt.tight_layout()
    plt.savefig('figures/data_exploration.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/data_exploration.png")

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/correlation_heatmap.png")

def preprocess_data(df):
    """Preprocess data for modeling"""
    print("\n" + "="*80)
    print("STEP 3: DATA PREPROCESSING")
    print("="*80)

    # Convert wine_type to numerical
    df['wine_type_encoded'] = (df['wine_type'] == 'red').astype(int)

    # Select features (exclude target and original categorical columns)
    feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'wine_type_encoded']

    X = df[feature_cols]
    y = df['quality_binary']

    print(f"\n✓ Features selected: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")
    print(f"\n✓ Target variable: quality_binary (binary classification)")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✓ Data split completed")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n✓ Feature scaling completed (StandardScaler)")

    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved: models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def train_models(X_train, y_train):
    """Train all machine learning models"""
    print("\n" + "="*80)
    print("STEP 4: MODEL TRAINING")
    print("="*80)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"✓ Model trained successfully")
        print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Save model
        model_filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved: {model_filename}")

    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and compare performance"""
    print("\n" + "="*80)
    print("STEP 5: MODEL EVALUATION")
    print("="*80)

    results = []

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"{name.upper()}")
        print('='*60)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Poor Wine', 'Good Wine']))

        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\n✓ Saved: results/model_comparison.csv")

    return results_df

def visualize_results(results_df, models, X_test, y_test):
    """Create visualizations for model comparison"""
    print("\n" + "="*80)
    print("STEP 6: RESULTS VISUALIZATION")
    print("="*80)

    # Model comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, color=color, legend=False)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/model_comparison.png")

    # Confusion matrices for all models
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Poor', 'Good'], yticklabels=['Poor', 'Good'])
        axes[idx].set_title(f'{name} - Confusion Matrix', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/confusion_matrices.png")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*15 + "WINE QUALITY PREDICTION - ML PROJECT")
    print("="*80)

    # Load and explore data
    df = load_and_explore_data('data/wine_quality.csv')

    # Visualize data
    visualize_data(df)

    # Preprocess data
    X_train, X_test, y_train, y_test, feature_cols = preprocess_data(df)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate models
    results_df = evaluate_models(trained_models, X_test, y_test)

    # Visualize results
    visualize_results(results_df, trained_models, X_test, y_test)

    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  • data/wine_quality.csv - Dataset")
    print("  • models/*.pkl - Trained models and scaler")
    print("  • results/model_comparison.csv - Performance metrics")
    print("  • figures/*.png - Visualizations")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
