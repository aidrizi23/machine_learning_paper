"""
Simple prediction script for Iris flower classification
Input flower measurements and get species prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model():
    """Train a Random Forest model on full dataset"""
    df = pd.read_csv('data/iris.csv')
    X = df[['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']]
    y = df['species']

    # Train on 80% of data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """Predict iris species from measurements"""

    # Create input dataframe
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    # Train model
    model = train_model()

    # Predict
    prediction = model.predict(input_data)[0]

    # Map to species name
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    species_name = species_map[prediction]

    return species_name

if __name__ == "__main__":
    print("=" * 50)
    print("IRIS FLOWER SPECIES PREDICTOR")
    print("=" * 50)

    # Example predictions
    examples = [
        (5.1, 3.5, 1.4, 0.2, "Setosa"),
        (6.7, 3.1, 4.7, 1.5, "Versicolor"),
        (7.2, 3.6, 6.1, 2.5, "Virginica")
    ]

    print("\nExample Predictions:\n")

    for sepal_l, sepal_w, petal_l, petal_w, expected in examples:
        prediction = predict_iris(sepal_l, sepal_w, petal_l, petal_w)
        print(f"Input: Sepal L={sepal_l}, Sepal W={sepal_w}, "
              f"Petal L={petal_l}, Petal W={petal_w}")
        print(f"Prediction: {prediction} (Expected: {expected})")
        print()

    print("=" * 50)
    print("\nTo use this predictor in your code:")
    print("from predict import predict_iris")
    print("result = predict_iris(5.1, 3.5, 1.4, 0.2)")
    print("print(result)  # Output: 'Setosa'")
