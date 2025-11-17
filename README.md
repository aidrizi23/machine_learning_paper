# Iris Flower Classification - Machine Learning Project

A simple beginner-friendly machine learning project comparing four classification algorithms on the famous Iris dataset.

## 🎯 Project Overview

This project demonstrates how to use machine learning to classify iris flowers into three species based on their physical measurements. It's perfect for beginners learning machine learning concepts.

## 📊 Dataset

**Iris Dataset** - 150 samples, 4 features, 3 classes
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width
- Classes: Setosa, Versicolor, Virginica
- Perfect balance: 50 samples per class

## 🤖 Machine Learning Models

Four algorithms are implemented, each in its own file:

1. **Logistic Regression** (`model_logistic_regression.py`)
2. **Decision Tree** (`model_decision_tree.py`)
3. **Random Forest** (`model_random_forest.py`)
4. **Support Vector Machine** (`model_svm.py`)

## 📁 Project Structure

```
machine_learning_paper/
├── data/
│   └── iris.csv                    # Dataset
├── output/
│   ├── results.csv                 # Model comparison results
│   └── ML_Project_Paper.docx       # Complete research paper
├── prepare_data.py                 # Download and prepare dataset
├── model_logistic_regression.py    # Logistic Regression implementation
├── model_decision_tree.py          # Decision Tree implementation
├── model_random_forest.py          # Random Forest implementation
├── model_svm.py                    # SVM implementation
├── run_all_models.py              # Train and compare all models
├── predict.py                      # Prediction script for new data
├── generate_paper.py               # Generate DOCX paper
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python prepare_data.py
```

### 3. Train All Models

```bash
python run_all_models.py
```

### 4. Make Predictions

```bash
python predict.py
```

### 5. Generate Paper

```bash
python generate_paper.py
```

## 📝 Results

All four models achieved **100% accuracy** on the test set!

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 100% |
| Decision Tree | 100% |
| Random Forest | 100% |
| SVM | 100% |

## 🔮 Using the Prediction Function

```python
from predict import predict_iris

# Predict species for new flower measurements
species = predict_iris(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2
)

print(f"Predicted species: {species}")
# Output: Predicted species: Setosa
```

## 📄 Research Paper

A complete beginner-level research paper is automatically generated in DOCX format:
- **File**: `output/ML_Project_Paper.docx`
- **Contents**: Abstract, Introduction, Methodology, Results, Discussion, Conclusion

The paper includes:
- Background and objectives
- Dataset description
- Explanation of each algorithm
- Performance comparison
- Practical applications
- Future work suggestions

## 🎓 Learning Objectives

This project teaches:
- ✅ Loading and preparing datasets
- ✅ Training machine learning models
- ✅ Evaluating model performance
- ✅ Comparing different algorithms
- ✅ Making predictions on new data
- ✅ Writing technical documentation

## 📚 Requirements

- Python 3.11+
- pandas 2.0.3
- numpy 1.24.3
- scikit-learn 1.3.0
- matplotlib 3.7.2
- python-docx 1.1.0

## 🤝 Perfect For

- Machine learning beginners
- University coursework
- Learning algorithm comparison
- Understanding classification tasks
- Portfolio projects

## 📖 Key Concepts Covered

1. **Data Preprocessing**: Loading, splitting, scaling
2. **Model Training**: Fitting models to training data
3. **Model Evaluation**: Testing on unseen data
4. **Algorithm Comparison**: Understanding strengths and weaknesses
5. **Prediction**: Using trained models for new data

## 🎯 Why Iris Dataset?

The Iris dataset is perfect for beginners because:
- Small and manageable (150 samples)
- Clean data (no missing values)
- Well-separated classes (easy to classify)
- Fast training (seconds)
- Famous in ML education

## 🔧 Customization

You can easily extend this project:
- Try different datasets
- Add more algorithms
- Tune hyperparameters
- Add visualizations
- Create a web interface

## 📞 Support

If you encounter issues:
1. Make sure all dependencies are installed
2. Run scripts in the order listed above
3. Check that `data/iris.csv` exists
4. Verify Python version is 3.11+

## 📜 License

Educational project - free to use and modify for learning purposes.

---

**Made for Machine Learning Beginners** 🎓
