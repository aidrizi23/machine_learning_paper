"""
Decision Tree Model for Iris Classification
Creates a tree of if-then rules to make decisions
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_decision_tree():
    """Train and evaluate Decision Tree"""
    print("=" * 50)
    print("DECISION TREE")
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

    # Train model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {'name': 'Decision Tree', 'accuracy': accuracy}

if __name__ == "__main__":
    train_decision_tree()
