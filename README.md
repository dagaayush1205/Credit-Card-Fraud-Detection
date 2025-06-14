# Credit-Card-Fraud-Detection
## Fraud Detection Using XGBoost

This project implements a robust **fraud detection system** using the **XGBoost classifier**, optimized through **GridSearchCV**, and evaluated using **AUPRC** (Area Under the Precision-Recall Curve), which is ideal for highly imbalanced classification problems like credit card fraud.

```
## Project Structure
├── data
│   └── creditcard.csv
├── detect_anomalies.py
├── eval.py
├── evaluate.py
├── models
│   ├── autoencoder.h5
│   └── isolation_forest.pkl
├── output
│   └── anomalies.json
├── preprocess.py
├── README.md
├── requirements.txt
├── train_model.py
├── train_xgboost.py
├── train_xgboost_split.py
└── utils
    ├── evaluate.py
```
the dataset has been taken from [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```
---
Traina and validate
```
python train_xgboost_split.py
```
