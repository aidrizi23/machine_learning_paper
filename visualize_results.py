"""
Results Visualization Module
Creates plots and charts for model comparison
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations():
    """
    Create visualizations for model comparison
    """
    # Load results
    results_df = pd.read_csv('results/model_comparison.csv')

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Machine Learning Model Comparison', fontsize=16, fontweight='bold')

    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if acc == results_df['Accuracy'].max() else '#3498db'
              for acc in results_df['Accuracy']]
    ax1.bar(results_df['Model'], results_df['Accuracy'], color=colors)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([0.9, 1.01])
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['Accuracy']):
        ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 2. Training Time Comparison
    ax2 = axes[0, 1]
    colors = ['#e74c3c' if t == results_df['Training Time (s)'].max() else '#9b59b6'
              for t in results_df['Training Time (s)']]
    ax2.bar(results_df['Model'], results_df['Training Time (s)'], color=colors)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['Training Time (s)']):
        ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. All Metrics Comparison (Grouped Bar Chart)
    ax3 = axes[1, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = range(len(results_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax3.bar([p + offset for p in x], results_df[metric], width,
                label=metric, alpha=0.8)

    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax3.legend(loc='lower right')
    ax3.set_ylim([0.9, 1.01])

    # 4. Performance vs Speed Trade-off
    ax4 = axes[1, 1]
    scatter = ax4.scatter(results_df['Training Time (s)'], results_df['Accuracy'],
                          s=200, alpha=0.6, c=range(len(results_df)), cmap='viridis')
    ax4.set_xlabel('Training Time (seconds)', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Performance vs Speed Trade-off', fontsize=12, fontweight='bold')

    for i, model in enumerate(results_df['Model']):
        ax4.annotate(model.split()[0],
                     (results_df['Training Time (s)'].iloc[i],
                      results_df['Accuracy'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()

    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to results/model_comparison.png")

    # Create a summary table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    create_visualizations()
