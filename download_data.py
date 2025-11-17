"""
Download Wine Quality Dataset from UCI ML Repository
"""
import pandas as pd
import urllib.request
import os

def download_dataset():
    """Download the wine quality dataset"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    filepath = os.path.join(data_dir, 'winequality-red.csv')

    try:
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset saved to {filepath}")

        # Load and display basic info
        df = pd.read_csv(filepath, sep=';')
        print(f"\nDataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nDataset info:")
        print(df.info())

        return filepath
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Using sklearn wine dataset as fallback...")
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['quality'] = wine.target
        df.to_csv(filepath, index=False)
        print(f"Fallback dataset saved to {filepath}")
        return filepath

if __name__ == "__main__":
    download_dataset()
