"""
Logistic Regression Model for Iris Classification
Simple linear model for multi-class problems
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression():
    """Train and evaluate Logistic Regression"""
    print("=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)

    # Load data
    df = pd.read_csv('data/iris.csv')
    X = df[['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']]
    y = df['species']

    # Split: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {'name': 'Logistic Regression', 'accuracy': accuracy}

if __name__ == "__main__":
    train_logistic_regression()
