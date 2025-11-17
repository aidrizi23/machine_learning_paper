"""
Run all models and compare their accuracy
"""

from model_logistic_regression import train_logistic_regression
from model_decision_tree import train_decision_tree
from model_random_forest import train_random_forest
from model_svm import train_svm
import pandas as pd

def compare_all_models():
    """Train all models and compare results"""
    print("\n" + "=" * 60)
    print(" " * 15 + "IRIS CLASSIFICATION PROJECT")
    print("=" * 60 + "\n")

    # Train all models
    results = []

    print("\n📊 Training Model 1/4...")
    results.append(train_logistic_regression())

    print("\n📊 Training Model 2/4...")
    results.append(train_decision_tree())

    print("\n📊 Training Model 3/4...")
    results.append(train_random_forest())

    print("\n📊 Training Model 4/4...")
    results.append(train_svm())

    # Create comparison table
    print("\n" + "=" * 60)
    print(" " * 20 + "FINAL RESULTS")
    print("=" * 60)

    df_results = pd.DataFrame(results)
    df_results['accuracy'] = df_results['accuracy'] * 100
    df_results = df_results.sort_values('accuracy', ascending=False)
    df_results.columns = ['Model', 'Accuracy (%)']

    print("\n" + df_results.to_string(index=False))

    # Save results
    df_results.to_csv('output/results.csv', index=False)
    print("\n✓ Results saved to output/results.csv")

    # Find best model
    best = df_results.iloc[0]
    print(f"\n🏆 Best Model: {best['Model']} with {best['Accuracy (%)']:.2f}% accuracy")

    return df_results

if __name__ == "__main__":
    compare_all_models()
