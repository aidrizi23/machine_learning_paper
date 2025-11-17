"""
Support Vector Machine (SVM) for Iris Classification
Finds the best boundary to separate different classes
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_svm():
    """Train and evaluate SVM"""
    print("=" * 50)
    print("SUPPORT VECTOR MACHINE (SVM)")
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

    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {'name': 'SVM', 'accuracy': accuracy}

if __name__ == "__main__":
    train_svm()
