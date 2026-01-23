# Bank Transfer Fraud Detection – IEEE-CIS Kaggle Competition
![Project Banner](../images/ieee_logo.png)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

This repository contains a comprehensive **Exploratory Data Analysis (EDA)** and **machine learning solution** for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) Kaggle competition.  
The goal is to predict whether an online transaction is fraudulent (`isFraud`) using real-world e-commerce transaction data provided by Vesta Corporation.

The notebook performs deep EDA, smart feature engineering, careful leakage-free preprocessing, model comparison, hyperparameter optimization with Optuna, and final LightGBM training — achieving strong validation **PR-AUC ~0.575**.

---
## Project Overview

Online payment fraud causes massive financial losses every year. This project analyzes anonymized transaction and identity data to uncover fraud patterns and build a reliable classifier.

### Tools & Libraries used
- **Pandas**, **NumPy** — data manipulation  
- **Seaborn**, **Matplotlib**, **Altair** — static & interactive visualizations  
- **Scikit-learn** — preprocessing, metrics, pipelines  
- **LightGBM**, **XGBoost**, **CatBoost** — gradient boosting models  
- **Optuna** — hyperparameter tuning  
- **Joblib** — model & preprocessor persistence  

**Key components:**
- Datasets: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`
- Main Notebook: `Transaction_Fraud_Detection_ETA_ML.ipynb`

---
## Objectives
- Understand fraud rate (~3.5%) and severe class imbalance
- Analyze missingness patterns (especially identity table — only ~24% coverage)
- Discover high-risk segments (product codes, email domains, device types, amounts, time-of-day)
- Engineer safe, leakage-free features (device risk encoding, email domain grouping, row-wise statistics)
- Build and tune gradient boosting models with focus on **PR-AUC** (suitable for heavy imbalance)
- Create production-ready inference bundle (preprocessor + model)

---
## Dataset
**Source:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)  
**Size:**  
• train_transaction: 590,540 rows × 394 columns  
• train_identity:   144,233 rows × 41 columns  

**Main files:**
- `train_transaction.csv` — core transaction features + target `isFraud`
- `train_identity.csv`    — device, browser, IP-related identity info (sparse)
- Test files follow the same structure (no target)

**Important notes:**
- Extremely wide dataset (hundreds of anonymized Vesta features — `V1`–`V339`)
- Heavy missing values, especially in identity & some V columns
- High-cardinality categoricals (emails, device info, card IDs)
- Time feature `TransactionDT` — timedelta from unknown reference

---
## Notebook Structure
1. Libraries & Data Loading  
2. Basic checks (shape, types, missing, duplicates)  
3. Deep EDA  
   - Fraud rate & imbalance visualization  
   - Missingness comparison (with vs without identity info)  
   - High-risk segments (product, card, email, device)  
   - Amount, time, and categorical fraud patterns  
4. Feature Engineering (leakage-safe)  
   - Email domain extraction & grouping  
   - Device risk target encoding (train-only fit)  
   - Row-wise statistics, boolean conversions  
5. Preprocessing Pipeline (sanitization → encoding → scaling)  
6. Model Training & Tuning  
   - Baseline → LightGBM / XGBoost / CatBoost comparison  
   - Optuna tuning (PR-AUC optimization, 60 min budget)  
   - Time-based train/validation split  
7. Final Model & Inference Bundle  
   - Retrain on 100% data with best iteration  
   - Save preprocessor + model (`lgbm_bundle.joblib`)  
8. Test predictions & threshold selection

---
## Key Findings
- Only ~3.5% of transactions are fraudulent → strong imbalance  
- Transactions **with identity information** show significantly **lower fraud rate**  
- Certain **ProductCD** values (especially 'C' & 'R') and some email providers are much riskier  
- **DeviceInfo** fingerprinting + rare devices → strong fraud signal  
- Very high **TransactionAmt** outliers appear more often in fraud  
- Missing identity data itself is predictive (absence correlates with higher fraud)  
- Strongest engineered feature: **device-level target-encoded risk**

![Example Visual – Fraud by ProductCD](./images/fraud_by_productcd_example.png)  
*(more visuals inside notebook — correlation heatmaps, boxplots, time patterns, etc.)*

---
## Machine Learning Model
**Final choice:** LightGBM  
**Validation metric (focus):** PR-AUC ≈ **0.575** (strong improvement over baseline ~0.03–0.05)  
**Best parameters found via Optuna:**
```text
num_leaves:        94
max_depth:         9
learning_rate:     0.0714
colsample_bytree:  0.860
subsample:         0.926
reg_alpha:         0.001
reg_lambda:        0.558
min_child_samples: 65
n_estimators:      583 (early stopping)
