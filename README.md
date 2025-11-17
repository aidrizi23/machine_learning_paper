# Wine Quality Prediction - Machine Learning Project

A comprehensive machine learning project comparing multiple algorithms for wine quality prediction based on physicochemical properties.

## Project Overview

This project implements and compares four machine learning algorithms for predicting wine quality:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

The study analyzes a dataset of 2,000 wine samples with 11 physicochemical features to predict binary quality classification (good vs. poor wines).

## Project Structure

```
machine_learning_paper/
├── data/
│   └── wine_quality.csv          # Generated wine quality dataset
├── models/
│   ├── logistic_regression.pkl   # Trained Logistic Regression model
│   ├── decision_tree.pkl         # Trained Decision Tree model
│   ├── random_forest.pkl         # Trained Random Forest model
│   ├── svm.pkl                   # Trained SVM model
│   └── scaler.pkl                # Feature scaler
├── results/
│   └── model_comparison.csv      # Performance metrics comparison
├── figures/
│   ├── data_exploration.png      # Exploratory data analysis plots
│   ├── correlation_heatmap.png   # Feature correlation matrix
│   ├── model_comparison.png      # Model performance comparison
│   └── confusion_matrices.png    # Confusion matrices for all models
├── generate_wine_data.py         # Dataset generation script
├── train_models.py               # Model training and evaluation pipeline
├── paper.md                      # Complete research paper
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd machine_learning_paper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas (2.3.3)
- numpy (2.3.5)
- scikit-learn (1.7.2)
- matplotlib (3.10.7)
- seaborn (0.13.2)

## Usage

### Generate Dataset

```bash
python generate_wine_data.py
```

This creates a realistic wine quality dataset with 2,000 samples in `data/wine_quality.csv`.

### Train Models

```bash
python train_models.py
```

This script will:
1. Load and explore the dataset
2. Preprocess and scale features
3. Train all four ML models
4. Evaluate performance metrics
5. Generate visualizations
6. Save trained models and results

Expected runtime: < 2 minutes on standard CPU

## Dataset Description

The wine quality dataset contains 2,000 samples with the following features:

### Features (11 physicochemical properties)
1. **fixed_acidity** - Tartaric acid content (g/dm³)
2. **volatile_acidity** - Acetic acid content (g/dm³)
3. **citric_acid** - Citric acid content (g/dm³)
4. **residual_sugar** - Sugar remaining after fermentation (g/dm³)
5. **chlorides** - Salt content (g/dm³)
6. **free_sulfur_dioxide** - SO₂ in free form (mg/dm³)
7. **total_sulfur_dioxide** - Total SO₂ (mg/dm³)
8. **density** - Wine density (g/cm³)
9. **pH** - Acidity level
10. **sulphates** - Potassium sulphate content (g/dm³)
11. **alcohol** - Alcohol percentage by volume (%)

### Target Variables
- **quality** - Original quality score (3-8)
- **quality_binary** - Binary classification (0=Poor [≤5], 1=Good [>5])
- **wine_type** - Red or White wine

### Statistics
- Total samples: 2,000 (1,000 red, 1,000 white)
- Training set: 1,600 samples (80%)
- Test set: 400 samples (20%)
- Class distribution: 89% poor quality, 11% good quality

## Results Summary

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | CV Accuracy |
|-------|----------|-----------|--------|----------|-------------|
| Random Forest | **88.75%** | 0.00% | 0.00% | 0.00% | **89.00%** ±0.23% |
| Logistic Regression | **88.50%** | 37.50% | 6.82% | 11.54% | 88.75% ±0.40% |
| SVM | 85.75% | 19.05% | 9.09% | **12.31%** | 87.56% ±0.23% |
| Decision Tree | 84.75% | 16.00% | 9.09% | 11.59% | 85.12% ±1.27% |

### Key Findings

1. **Random Forest and Logistic Regression** achieved the highest accuracy (~89%)
2. **Class imbalance** significantly impacts model performance on minority class
3. **SVM** achieved the best balance between precision and recall
4. **Alcohol content and volatile acidity** are the most important features
5. All models show stable cross-validation performance

### Visualizations

All generated visualizations are saved in the `figures/` directory:
- Quality distribution and exploratory plots
- Feature correlation heatmap
- Model performance comparison charts
- Confusion matrices for all models

## Research Paper

A comprehensive research paper (`paper.md`) is included, covering:
- Abstract and introduction
- Literature review
- Detailed methodology
- Experimental setup
- Results and analysis
- Discussion of findings
- Conclusions and future work
- References

The paper is suitable for university-level coursework and provides in-depth analysis of the machine learning techniques applied.

## Key Insights

### Strengths
- Multiple algorithms compared on identical dataset
- Comprehensive evaluation metrics
- Reproducible results with fixed random seeds
- Well-documented code and methodology

### Limitations
- Significant class imbalance (89%-11%)
- Binary classification loses granularity
- Limited to physicochemical features only
- Dataset size modest for deep learning approaches

### Future Improvements
- Implement class balancing techniques (SMOTE, class weights)
- Try deep learning models (neural networks)
- Multi-class classification for full quality scale
- Incorporate sensory evaluation data
- Expand dataset with more high-quality wine samples

## Model Usage Example

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data
new_wine = pd.DataFrame({
    'fixed_acidity': [7.5],
    'volatile_acidity': [0.3],
    'citric_acid': [0.35],
    'residual_sugar': [5.0],
    'chlorides': [0.05],
    'free_sulfur_dioxide': [30.0],
    'total_sulfur_dioxide': [120.0],
    'density': [0.996],
    'pH': [3.2],
    'sulphates': [0.6],
    'alcohol': [11.0],
    'wine_type_encoded': [1]  # 1 for red, 0 for white
})

# Scale features
new_wine_scaled = scaler.transform(new_wine)

# Predict
prediction = model.predict(new_wine_scaled)
print(f"Predicted quality: {'Good' if prediction[0] == 1 else 'Poor'}")
```

## Contributing

This is an educational project. Feel free to:
- Experiment with different hyperparameters
- Try additional ML algorithms
- Implement class balancing techniques
- Extend the analysis with new visualizations

## License

This project is created for educational purposes.

## Acknowledgments

- Inspired by the UCI Wine Quality dataset
- Uses scikit-learn for ML implementations
- Visualization tools: matplotlib and seaborn

## Contact

For questions or feedback about this project, please refer to the research paper for detailed methodology and results.

---

**Project completed:** November 2025
**Machine Learning Course Project**
