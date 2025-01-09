# XGBM-LGBM
# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using machine learning models such as LightGBM and XGBoost. The dataset is preprocessed and evaluated with various metrics to determine model performance.

## Dataset

The project uses the Titanic dataset provided by [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic). It consists of two files:
- `titanic_train.csv`: Training dataset with passenger details and survival status.
- `titanic_test.csv`: Test dataset for predictions.

## Project Workflow

### 1. Data Preprocessing
- **Handling Missing Values**:
  - Filled missing `Age` values with the median.
  - Filled missing `Embarked` values with the mode.
- **Feature Encoding**:
  - Converted categorical columns (`Sex`, `Embarked`) into numerical using one-hot encoding.
- **Feature Removal**:
  - Dropped unnecessary columns: `Name`, `Ticket`, `Cabin`.

### 2. Model Training
- Models used:
  - **LightGBM** (`LGBMClassifier`)
  - **XGBoost** (`XGBClassifier`)
- The training dataset was split into training and validation sets using an 80-20 split.

### 3. Model Evaluation
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

### 4. Visualization
- A bar chart is created to compare the performance of LightGBM and XGBoost across the metrics.

## Requirements

Install the required Python libraries:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib
