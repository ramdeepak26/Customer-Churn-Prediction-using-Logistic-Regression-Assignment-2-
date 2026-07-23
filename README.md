# Customer-Churn-Prediction-using-Logistic-Regression

## Overview
This project builds a **Logistic Regression** model to predict whether a telecom
customer is likely to churn (leave the company), based on demographic information
and service usage patterns.

## Dataset
- **Name:** Telco Customer Churn
- **Source:** [Kaggle - blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 rows × 21 columns
- **Target variable:** `Churn` (Yes/No)

## Project Structure
```
├── Customer_Churn_Prediction_LogisticRegression.ipynb   # Main notebook (all 4 tasks, with outputs)
├── churn_analysis.py                                    # Standalone script version
├── WA_Fn-UseC_-Telco-Customer-Churn.csv                 # Dataset
├── confusion_matrix.png                                 # Confusion matrix plot
├── feature_importance.png                               # Top 10 feature coefficients plot
├── model_metrics.txt                                    # Saved evaluation metrics
└── README.md
```

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run
**Option 1 — Notebook**
```bash
jupyter notebook Customer_Churn_Prediction_LogisticRegression.ipynb
```
**Option 2 — Script**
```bash
python churn_analysis.py
```

## Approach

### 1. Data Understanding
- Loaded the dataset and inspected the first 5 records.
- Identified 4 numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`,
  `SeniorCitizen`), 15 categorical features, and the target variable `Churn`.

### 2. Data Preprocessing
- Found 11 missing values in `TotalCharges` (blank strings for new customers with
  `tenure = 0`); imputed them with 0.
- Dropped `customerID` (non-predictive identifier).
- Label-encoded all categorical variables and the target (`Yes`→1, `No`→0).
- Split data into 80% train / 20% test (stratified).
- Scaled numerical features with `StandardScaler`.

### 3. Model Development
- Trained a `LogisticRegression` model (scikit-learn) on all 19 features.
- Generated predictions and churn probabilities on the test set.

### 4. Model Evaluation
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.80  |
| Precision | 0.64  |
| Recall    | 0.55  |
| F1-Score  | 0.59  |

**Confusion Matrix**

|                | Predicted No | Predicted Yes |
|----------------|:------------:|:-------------:|
| **Actual No**  | 922          | 113           |
| **Actual Yes** | 169          | 205           |

## Key Observations
1. Accuracy (~80%) is inflated by class imbalance (~73% no-churn); recall on
   churners (55%) shows the model misses a significant share of actual churners.
2. Precision (64%) exceeds recall (55%) — flagged churn predictions are fairly
   reliable, but the model is conservative and under-predicts the churn class.
3. `Contract` type, `tenure`, and charge amounts are the strongest churn drivers:
   short tenure, month-to-month contracts, and no security/support add-ons raise
   churn risk, while longer contracts and add-on services correlate with
   retention.
