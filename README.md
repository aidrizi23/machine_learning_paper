# Machine Learning Model Comparison Project

A comprehensive comparative study of four machine learning algorithms (SVM, Decision Trees, Random Forest, and Logistic Regression) for wine classification.

## Project Overview

This project implements and compares four classical machine learning algorithms on the Wine dataset from scikit-learn. The study includes data preprocessing, model training, performance evaluation, and a complete research paper documenting the methodology and findings.

## Results Summary

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 100.00% | 0.0996s |
| SVM | 97.22% | 0.0015s |
| Logistic Regression | 97.22% | 0.0076s |
| Decision Tree | 94.44% | 0.0014s |

## Project Structure

```
machine_learning_paper/
├── data/                  # Dataset directory
├── models/                # Saved trained models (.pkl files)
├── results/               # Results and visualizations
├── download_data.py       # Dataset download script
├── preprocess.py          # Data preprocessing module
├── train_models.py        # Model training and evaluation
├── visualize_results.py   # Results visualization
├── requirements.txt       # Python dependencies
├── research_paper.md      # Complete research paper
└── README.md             # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Run All Steps)

```bash
# 1. Download dataset
python3 download_data.py

# 2. Train all models
python3 train_models.py

# 3. Generate visualizations
python3 visualize_results.py
```

### Individual Scripts

**Download Data:**
```bash
python3 download_data.py
```

**Preprocess Data:**
```bash
python3 preprocess.py
```

**Train Models:**
```bash
python3 train_models.py
```

**Create Visualizations:**
```bash
python3 visualize_results.py
```

## Models Implemented

1. **Support Vector Machine (SVM)**
   - Kernel: RBF
   - Accuracy: 97.22%
   - Fastest training among high-accuracy models

2. **Decision Tree**
   - Max depth: 10
   - Accuracy: 94.44%
   - Most interpretable model

3. **Random Forest**
   - 100 estimators
   - Accuracy: 100%
   - Best overall performance

4. **Logistic Regression**
   - Multi-class: multinomial
   - Accuracy: 97.22%
   - Good probabilistic interpretation

## Key Findings

- **Random Forest** achieved perfect classification accuracy (100%)
- **SVM** provides the best accuracy-speed trade-off (97.22% accuracy, 0.0015s training)
- **Decision Tree** is fastest but least accurate (94.44% accuracy, 0.0014s training)
- **Logistic Regression** matches SVM accuracy with slightly longer training time

## Research Paper

A complete university-level research paper is included in `research_paper.md`, containing:
- Abstract and introduction
- Literature review
- Detailed methodology
- Comprehensive results and analysis
- Discussion of findings
- Conclusions and future work
- References

## Dependencies

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Dataset

The project uses the Wine dataset from scikit-learn, which contains:
- 178 samples
- 13 chemical features
- 3 classes (wine types)
- No missing values

## Reproducibility

All experiments use `random_state=42` for reproducibility. To reproduce results exactly, ensure you're using compatible versions of scikit-learn and other dependencies.

## License

This project is for educational purposes.

## Author

Machine Learning Comparative Study Project

## Citation

If you use this code or findings, please reference the included research paper.
