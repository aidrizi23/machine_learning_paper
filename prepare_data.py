"""
Download and prepare the Iris dataset for machine learning
Simple dataset perfect for beginners: 150 samples, 4 features, 3 classes
"""

from sklearn.datasets import load_iris
import pandas as pd

def prepare_iris_data():
    """Download and save Iris dataset"""
    print("Loading Iris dataset...")

    # Load the famous Iris dataset
    iris = load_iris()

    # Create DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Save to CSV
    df.to_csv('data/iris.csv', index=False)

    print(f"✓ Dataset saved to data/iris.csv")
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {list(iris.feature_names)}")
    print(f"  Classes: {list(iris.target_names)}")
    print(f"\nClass distribution:")
    print(df['species_name'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())

    return df

if __name__ == "__main__":
    prepare_iris_data()
