"""
Download Wine Quality Dataset
This script downloads the wine quality dataset from a reliable source.
"""

import pandas as pd
import os

def download_wine_data():
    """Download wine quality dataset from UCI repository using pandas"""

    print("Downloading wine quality dataset...")

    # Try multiple sources
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    ]

    for url in urls:
        try:
            # Red wine dataset
            df_red = pd.read_csv(url, sep=';')
            df_red['wine_type'] = 'red'
            print(f"✓ Downloaded red wine data: {len(df_red)} samples")

            # White wine dataset
            url_white = url.replace('red', 'white')
            df_white = pd.read_csv(url_white, sep=';')
            df_white['wine_type'] = 'white'
            print(f"✓ Downloaded white wine data: {len(df_white)} samples")

            # Combine datasets
            df = pd.concat([df_red, df_white], axis=0, ignore_index=True)

            # Save to CSV
            output_path = 'data/wine_quality.csv'
            df.to_csv(output_path, index=False)
            print(f"\n✓ Dataset saved to {output_path}")
            print(f"Total samples: {len(df)}")
            print(f"Features: {list(df.columns)}")
            print(f"\nDataset info:")
            print(df.info())
            print(f"\nQuality distribution:")
            print(df['quality'].value_counts().sort_index())

            return True

        except Exception as e:
            print(f"Failed with {url}: {e}")
            continue

    # If download fails, try fetching from OpenML
    print("\nDownload failed. Trying OpenML repository...")
    try:
        from sklearn.datasets import fetch_openml

        # Fetch wine quality dataset from OpenML
        wine_data = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='auto')
        df_red = wine_data.frame
        df_red['wine_type'] = 'red'

        print(f"✓ Downloaded red wine data from OpenML: {len(df_red)} samples")

        # Try to get white wine too
        try:
            wine_data_white = fetch_openml(name='wine-quality-white', version=1, as_frame=True, parser='auto')
            df_white = wine_data_white.frame
            df_white['wine_type'] = 'white'
            print(f"✓ Downloaded white wine data from OpenML: {len(df_white)} samples")

            df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
        except:
            df = df_red
            print("Could not fetch white wine data, using red wine only")

        output_path = 'data/wine_quality.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dataset saved to {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Features: {list(df.columns)}")
        return True

    except Exception as e:
        print(f"OpenML fetch also failed: {e}")
        print("\nUsing sklearn wine dataset as final fallback...")
        from sklearn.datasets import load_wine

        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['quality'] = wine.target
        df['wine_type'] = 'red'  # Default type

        output_path = 'data/wine_quality.csv'
        df.to_csv(output_path, index=False)
        print(f"✓ Fallback dataset saved to {output_path}")
        print(f"Total samples: {len(df)}")

        return True

if __name__ == "__main__":
    download_wine_data()
