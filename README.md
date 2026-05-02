# Comment Prediction Analysis

A comprehensive machine learning project for predicting comment categories using advanced classification algorithms. This project analyzes comments and predicts their categories through exploratory data analysis, feature engineering, and multi-model evaluation.

## Overview

This project participates in the Kaggle Comment Category Prediction Challenge, implementing an end-to-end machine learning pipeline that includes:
- Data exploration and statistical analysis
- Feature engineering and text preprocessing
- Multiple classifier implementations (Logistic Regression, XGBoost, LightGBM, SVM, KNN, Random Forest, and more)
- Hyperparameter tuning via RandomizedSearchCV
- Model evaluation and comparison
- Handling of class imbalance with balanced ensemble methods

## Features

✨ **Key Capabilities:**
- Comprehensive exploratory data analysis (EDA) with visualization
- Text processing and TF-IDF vectorization
- Multiple machine learning algorithms for classification
- Class imbalance handling using `imbalanced-learn`
- Feature selection and dimensionality reduction
- Robust model evaluation with multiple metrics
- Hyperparameter optimization
- Pipeline-based preprocessing for reproducibility

## Requirements

### Python Version
- Python 3.7+

### Dependencies
```
numpy                  # Linear algebra operations
pandas                 # Data manipulation and analysis
matplotlib             # Data visualization
seaborn                # Statistical visualization
scipy                  # Scientific computing
scikit-learn           # Machine learning library
xgboost                # Gradient boosting
lightgbm               # Light gradient boosting
imbalanced-learn       # Class imbalance handling
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Comment-Prediction-Analysis
```

### 2. Create a Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n comment-pred python=3.9
conda activate comment-pred
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset

The train and test data for this project is available at the Kaggle competition:
**[Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)**

You can download the datasets directly from the competition page and place them in the project directory.

## Usage

### Running the Analysis

1. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```

2. **Execute Cells Sequentially:**
   - Start from the top cell and run each cell in order
   - Each section performs specific analysis and preprocessing steps

3. **Key Sections in the Notebook:**
   - **Libraries & Setup**: Import all required libraries
   - **Data Loading**: Load training and test datasets
   - **Exploratory Data Analysis**: Visualize class distribution and feature relationships
   - **Data Preprocessing**: Handle missing values and encode categorical variables
   - **Feature Engineering**: Create and select important features
   - **Model Training**: Train multiple classifiers
   - **Model Evaluation**: Compare model performance
   - **Predictions**: Generate predictions on test data

### Expected Data Format

The project expects CSV files with the following structure:
- **Train Data** (`train.csv`):
  - `post_id`: Post identifier
  - `comment`: Text content of the comment
  - `created_date`: Timestamp of creation
  - Categorical features
  - Numerical features
  - `label`: Target variable (comment category)

- **Test Data** (`test.csv`):
  - Same columns as train data except `label`

## Project Structure

```
Comment-Prediction-Analysis/
├── notebook.ipynb          # Main Jupyter notebook with complete analysis pipeline
├── requirements.txt        # Python package dependencies
├── README.md              # Project documentation
└── plots/                 # Generated visualizations (created during execution)
```

### Notebook Sections Breakdown

| Section | Purpose |
|---------|---------|
| Libraries | Import all required ML and visualization libraries |
| Data Loading | Load train and test datasets |
| EDA | Analyze class distribution, missing values, feature correlations |
| Preprocessing | Clean data, handle missing values, encode features |
| Feature Engineering | Create new features, apply transformations |
| Model Training | Train multiple classifiers with various algorithms |
| Model Evaluation | Compare model performance using multiple metrics |
| Hyperparameter Tuning | Optimize model parameters using RandomizedSearchCV |
| Predictions | Generate final predictions on test set |

## Models Implemented

The project includes implementations of:

- **Linear Models**: Logistic Regression, Ridge Classifier, SGD Classifier
- **Probabilistic Models**: Naive Bayes (Multinomial, Complement)
- **Tree-based Models**: Random Forest, XGBoost, LightGBM
- **Distance-based Models**: K-Nearest Neighbors, Linear SVC
- **Ensemble Models**: Bagging Classifier, Balanced Bagging Classifier
- **Baseline**: Dummy Classifier (most frequent strategy)

## Key Techniques

### Feature Engineering
- TF-IDF vectorization for text features
- Categorical encoding (One-Hot, Target encoding)
- Feature scaling (StandardScaler, MinMaxScaler, MaxAbsScaler)
- Feature selection (SelectKBest, RFE, SelectFromModel)
- Dimensionality reduction (TruncatedSVD)

### Model Evaluation Metrics
- Accuracy Score
- F1 Score
- Classification Report (Precision, Recall, F1 per class)
- Confusion Matrix

### Handling Class Imbalance
- Balanced Bagging Classifier
- Stratified sampling in train-test split
- Class weight balancing in models

## Output

The notebook generates:
- Visualizations saved in the `plots/` directory
- Model performance reports
- Prediction CSV file (`submission.csv`) with format:
  ```
  ID,label
  1,category_0
  2,category_1
  ...
  ```

## Performance Considerations

- **Data Size**: The project handles datasets with thousands of comments efficiently
- **Computation Time**: Training multiple models may take several minutes depending on dataset size
- **Memory**: Approximately 2-4 GB RAM recommended for smooth execution
- **Storage**: Plot visualizations generate PNG files (~500 KB - 2 MB total)

## Tips for Best Results

1. **Data Quality**: Ensure input CSV files are properly formatted with no encoding issues
2. **Feature Selection**: Adjust `SelectKBest(k=...)` parameters based on your feature count
3. **Hyperparameter Tuning**: Modify `RandomizedSearchCV` n_iter for more exhaustive search
4. **Class Imbalance**: If severe, increase `n_samples_bootstrap` in Balanced Bagging
5. **Memory Optimization**: Reduce TF-IDF vocabulary size if memory-constrained

## Dependencies Details

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | Latest | Numerical operations |
| pandas | Latest | Data manipulation |
| matplotlib | Latest | Plotting and visualization |
| seaborn | Latest | Statistical visualization |
| scipy | Latest | Statistical functions |
| scikit-learn | Latest | ML algorithms and utilities |
| xgboost | Latest | Gradient boosting classifier |
| lightgbm | Latest | Light gradient boosting classifier |
| imbalanced-learn | Latest | Handling imbalanced datasets |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Ensure all packages are installed: `pip install -r requirements.txt` |
| Memory error | Reduce TF-IDF vocabulary or downsample data |
| Data not found | Verify Kaggle input path or update data location in code |
| Slow execution | Use LightGBM instead of XGBoost, reduce hyperparameter search space |
| Path errors (Windows) | Use raw strings (r"path") or forward slashes for file paths |

## Contributing

Contributions are welcome! To improve this project:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Kaggle for the Comment Category Prediction Challenge
- scikit-learn, XGBoost, and LightGBM communities for excellent ML libraries
- Kaggle Docker Python image for pre-configured environment

## Contact & Support

For questions or issues, please open an issue in the repository or refer to the inline documentation within the notebook.

---

**Last Updated**: May 2, 2026  
**Status**: Active Development
