"""
Generate Wine Quality Dataset
Since external sources are blocked, we'll generate a realistic wine quality dataset
based on the characteristics of the UCI Wine Quality dataset.
"""

import pandas as pd
import numpy as np

def generate_wine_dataset(n_samples=1500, random_state=42):
    """
    Generate a realistic wine quality dataset
    Based on the UCI Wine Quality dataset characteristics
    """
    np.random.seed(random_state)

    # Generate features with realistic correlations
    # Features: fixed acidity, volatile acidity, citric acid, residual sugar,
    # chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

    n_red = n_samples // 2
    n_white = n_samples - n_red

    def generate_wine_samples(n, wine_type):
        """Generate wine samples for a specific type"""

        if wine_type == 'red':
            # Red wine characteristics
            fixed_acidity = np.random.normal(8.3, 1.7, n).clip(4, 16)
            volatile_acidity = np.random.normal(0.53, 0.18, n).clip(0.1, 1.6)
            citric_acid = np.random.normal(0.27, 0.19, n).clip(0, 1)
            residual_sugar = np.random.gamma(2, 1.3, n).clip(0.5, 16)
            chlorides = np.random.normal(0.087, 0.047, n).clip(0.01, 0.6)
            free_sulfur_dioxide = np.random.gamma(3, 5, n).clip(1, 72)
            total_sulfur_dioxide = free_sulfur_dioxide * np.random.uniform(2, 8, n)
            density = 0.9967 + fixed_acidity * 0.0001 + residual_sugar * 0.00038 + np.random.normal(0, 0.002, n)
            pH = 3.3 - fixed_acidity * 0.03 + np.random.normal(0, 0.15, n)
            sulphates = np.random.normal(0.66, 0.17, n).clip(0.3, 2)
            alcohol = np.random.normal(10.4, 1.1, n).clip(8, 15)
        else:
            # White wine characteristics
            fixed_acidity = np.random.normal(6.9, 0.9, n).clip(3.8, 14.2)
            volatile_acidity = np.random.normal(0.28, 0.10, n).clip(0.08, 1.1)
            citric_acid = np.random.normal(0.33, 0.12, n).clip(0, 1.66)
            residual_sugar = np.random.gamma(3, 2, n).clip(0.6, 65)
            chlorides = np.random.normal(0.046, 0.022, n).clip(0.009, 0.35)
            free_sulfur_dioxide = np.random.gamma(5, 7, n).clip(2, 289)
            total_sulfur_dioxide = free_sulfur_dioxide * np.random.uniform(2.5, 6, n)
            density = 0.9940 + fixed_acidity * 0.0001 + residual_sugar * 0.00042 + np.random.normal(0, 0.003, n)
            pH = 3.2 - fixed_acidity * 0.02 + np.random.normal(0, 0.15, n)
            sulphates = np.random.normal(0.49, 0.11, n).clip(0.22, 1.08)
            alcohol = np.random.normal(10.5, 1.2, n).clip(8, 14.2)

        # Generate quality based on feature values (with realistic correlations)
        quality_score = (
            alcohol * 0.3 +
            volatile_acidity * (-5.0) +
            sulphates * 2.0 +
            citric_acid * 1.5 +
            fixed_acidity * 0.1 +
            residual_sugar * 0.05 +
            np.random.normal(0, 1.5, n)
        )

        # Map continuous score to discrete quality ratings (3-9)
        quality = np.round(4 + (quality_score - quality_score.mean()) / quality_score.std() * 1.2).astype(int)
        quality = quality.clip(3, 9)

        # Create DataFrame
        df = pd.DataFrame({
            'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol,
            'quality': quality,
            'wine_type': wine_type
        })

        return df

    # Generate both types
    df_red = generate_wine_samples(n_red, 'red')
    df_white = generate_wine_samples(n_white, 'white')

    # Combine
    df = pd.concat([df_red, df_white], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

if __name__ == "__main__":
    print("Generating wine quality dataset...")

    df = generate_wine_dataset(n_samples=2000, random_state=42)

    # Save to CSV
    output_path = 'data/wine_quality.csv'
    df.to_csv(output_path, index=False)

    print(f"✓ Dataset generated and saved to {output_path}")
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print(f"\nFeature statistics:")
    print(df.describe())
    print(f"\nQuality distribution:")
    print(df['quality'].value_counts().sort_index())
    print(f"\nWine type distribution:")
    print(df['wine_type'].value_counts())
