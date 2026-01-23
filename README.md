# Bank Transfer Fraud Detection – IEEE-CIS Kaggle Competition
![Project Banner](./images/ieee_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

This repository contains a comprehensive **Exploratory Data Analysis (EDA)** and **machine learning solution** for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) Kaggle competition.

The goal is to predict whether an online transaction is fraudulent (`isFraud`) using real-world e-commerce transaction data provided by Vesta Corporation.

The notebook performs deep EDA, feature engineering, careful leakage-free preprocessing, model comparison, hyperparameter optimization with Optuna, and final LightGBM training — achieving strong validation **PR-AUC ~0.575**.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Notebook Structure](#notebook-structure)
- [Key Findings](#key-findings)
- [Machine Learning Model](#machine-learning-model)
- [Feature Engineering Approach](feature-engineering-approach)
- [Ethical Considerations, Data Privacy & Governance](#ethical-considerations-data-privacy--governance)
- [Requirements](#requirements)
- [How to Reproduce](#how-to-reproduce)
- [Contributing](#contributing)
- [License](#license)
- [Credits and Acknowledgements](#credits-and-acknowledgements)
- [Author](#author)
  
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
- **Power Bi** - Quick Dataset overview Dashboard
- **Figma** - Dashboard layout design

**Key components:**
- Datasets: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` (not availiable on repository, download from Kaggle ~ 1Gb)
- Main Notebook: `Transaction_Fraud_Detection_ETA_ML.ipynb`
- Power Bi Dashborad: `dataset_overview.pbix` [open on server](https://app.powerbi.com/view?r=eyJrIjoiZjFhMWY4NGMtNzI3NC00ZmVhLThmODQtMjJmMjVmYTNmMjllIiwidCI6IjU5YTZhM2Y5LTMwYWItNDBmZi1hNDZhLWYzZThkZDU4OGZhOSIsImMiOjl9)
---
## Objectives
- Understand fraud rate (~3.5%) and severe class imbalance
- Analyze missingness patterns (especially identity table — only ~24% coverage)
- Discover high-risk segments (product codes, email domains, device types, amounts, time-of-day)
- Engineer safe, leakage-free features (device risk encoding, email domain grouping, row-wise statistics)
- Build and tune ML models with focus on **PR-AUC** (suitable for heavy imbalance)
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
- Certain **ProductCD** values (especially 'C' & 'R') and some email providers are much riskier
- The dataset exhibits a high degree of missingness, affecting the majority of features
- Missing identity data itself is predictive
- **DeviceInfo** fingerprinting + rare devices → strong fraud signal   
- Strongest engineered feature: **device-level target-encoded risk**
- Mac devices were the most free of fraud. "Others", "unknown" and Android are the most fraudulent, but no extreme skeve
- Screen sizes 1024x600 and 0x0 have the most Fraud risk
- The ‘Outlook’ email provider and the ‘.se’ and ‘.com’ domains are the most fraud-related
- Missing receiver email information is not a predictor of fraud

  #### Few visuals from the notebook attached below


![Example Visual – Fraud by ProductCD](./images/img1.png)

![Example Visual – Fraud by ProductCD](./images/img2.png)  

![Example Visual – Fraud by ProductCD](./images/img3.png)  
*(more visuals inside notebook — correlation heatmaps, boxplots etc.)*

---
## Machine Learning Model
**Final choice:** LightGBM  
**Validation metric (focus):** PR-AUC ≈ **0.575** (strong improvement over baseline ~0.03)  
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
```
Inference is bundled into one .joblib file containing:
* Device risk encoder (target encoding fitted on train)
* Full sklearn pipeline (sanitizer → converters → engineer → preprocessor)
* Trained LightGBM model
  
---

## Feature Engineering Approach

Feature engineering was one of the most critical steps in improving model performance on this highly anonymized, wide, and noisy dataset. All transformations were designed to be leakage-free, especially important given the time-series nature of transactions (`TransactionDT`) and the severe class imbalance.

### Guiding Principles

- Avoid any target leakage: target encoding and frequency maps fitted only on training data
- Respect temporal order: time-based split used for validation; encoders fitted only on train fold
- Handle high missingness intelligently rather than simple imputation
- Exploit domain knowledge from public kernels (email domains, device fingerprints, resolution parsing)
- Create interpretable + high-signal features while keeping cardinality manageable

### Core Custom Transformers

1. **DeviceRiskEncoder**

   - Target encoding of `DeviceInfo` column (smoothed Bayesian mean fraud rate per unique device string)
   - Fitted exclusively on training data (using alpha/beta smoothing with global fraud rate fallback)
   - Produces: `DeviceInfo_risk` (encoded fraud probability)
   - Impact: one of the strongest signals — rare/unknown devices strongly correlate with fraud

2. **TFBooleanConverter**

   - Converts M1–M9 match columns ("T" / "F" / NaN → 1.0 / 0.0 / NaN)
   - Automatically detects columns containing only {"T", "F"} (non-null)
   - Simple but necessary to make these categorical match fields numeric for boosting

3. **RowFeatureEngineer**

   - **DeviceInfo**
     - `DeviceInfo_freq` — normalized frequency (from train fit)
     - `DeviceInfo_is_missing` — binary flag (1 if NaN)

   - **Browser/OS (id_30, id_31)**
     - `id_30_os_family` — extracted OS family: "windows", "mac", "ios", "android", "linux" or "unknown"
     - `id_31_freq` — normalized frequency of id_31 values
     - `id_31_is_missing` — binary flag (1 if NaN)

   - **Screen resolution (id_33)**
     - `id_33_width`, `id_33_height` — numeric parsed values
     - `id_33_is_invalid` — flag (1 if exactly "0x0")

   - **Match status (id_34)**
     - `id_34_ord` — numeric value extracted from "match_status:XX" pattern

   - **Email domains (P_emaildomain, R_emaildomain)**
     - `{P,R}_emaildomain_is_missing` — binary flag (1 if NaN)
     - `{P,R}_emaildomain_domain` / `{P,R}_emaildomain_provider` — root part before first dot
     - `{P,R}_emaildomain_tld` — TLD part after first dot
     - `{P,R}_emaildomain_provider_freq` — normalized frequency (from train fit)

   - **Identity presence**
     - `has_identity` — 1 if `DeviceType` is not NaN (proxy for identity block presence)

   - **Missing ratios**
     - `missing_ratio_V` — mean missing in V1–V339 columns
     - `missing_ratio_D` — mean missing in D1–D15 columns
     - `missing_ratio_id` — mean missing in id_* columns
     - `high_missing_any` — 1 if any ratio > 0.7

   - **Cleanup**: drops original raw high-cardinality columns (`DeviceInfo`, `id_30`, `id_31`, `id_33`, `id_34`, `P_emaildomain`, `R_emaildomain`)

4. **Preprocessing Pipeline (build_preprocessor)**

   - Numeric columns → median imputation + StandardScaler
   - Categorical columns → most frequent imputation + OneHotEncoder (with min_frequency=50 for rare category grouping)
   - All wrapped in ColumnTransformer for selective processing

### Leakage & Time Protections

- DeviceRiskEncoder and frequency maps fitted only on train (`.fit(X_tr_raw, y_tr)`)
- Time-based split: first 80% train, last 20% validation
- No global/target statistics computed on full (train+test) data
- Test/validation transformation uses `.transform()` only (no refit)

### Most Valuable Engineered Features

- `DeviceInfo_risk` — dominant fraud signal
- Missing identity indicators (`has_identity`, `missing_ratio_id`, `high_missing_any`)
- Email domain provider/TLD + their frequencies
- `id_33_is_invalid` and parsed resolution dimensions (unusual sizes like 0×0 linked to fraud)
- Row-level missingness aggregates (`missing_ratio_V`, `missing_ratio_D`)

### Preprocessing Pipeline Integration

All steps were encapsulated in a scikit-learn Pipeline:

- `PandasSanitizer` → clean column names, fix types
- `TFBooleanConverter` → T/F → 1.0/0.0
- `RowFeatureEngineer` → missing ratios, resolution parsing, email domains, OS family
- `ColumnTransformer` → different handling for numeric/categorical/missing
- Final scaling (StandardScaler for numerics)

This pipeline ensures reproducibility and prevents leakage when transforming test data.

### Impact

The strongest single contributors to PR-AUC improvement were:

- Device risk encoding (dominant signal)
- Missing identity & high-missing flags
- Email domain provider frequencies
- Resolution invalid flag & parsing

---
## Ethical Considerations, Data Privacy & Governance

### Ethical Issues, Data Privacy, and Governance in Project Methodology

This project uses the publicly available, anonymized IEEE-CIS Fraud Detection dataset from Kaggle (Vesta Corporation). Personal identifiers (names, exact IPs, card numbers, emails) have been removed, pseudonymized, or hashed.

**Main safeguards applied:**
- No re-identification attempts or external data linkage
- Strictly educational/research purpose — no real-world or commercial deployment
- Data minimization: only the public subset is used
- Full transparency: preprocessing, features, model choices, and limitations documented in the notebook
- Fairness consideration: PR-AUC metric and time-based validation to handle imbalance and temporal effects
- Governance: MIT open-source license, reproducible pipeline (`lgbm_bundle.joblib`), no decisions affecting real people

Although anonymized, behavioral and device fingerprints remain. Theoretical re-identification risk via external linkage keeps this project strictly academic.

### Legal and Social Implications

**Legal context**  
The dataset is governed by Kaggle’s terms, allowing research use but prohibiting reverse-engineering of personal information. While no direct legal obligations apply here, real-world systems using similar data would need to comply with:
- GDPR (EU) — pseudonymized data → purpose limitation, DPIA
- CCPA/CPRA (US) — consumer rights to know/delete/opt-out
- PCI DSS — payment security standards
- AML/CTF regulations — mandatory audits for banking fraud models

This notebook highlights the importance of bias audits, explainability, human oversight, and access controls in production.

**Social implications**  
**Positive:** Reduces financial crime, protects users/merchants, lowers costs, enables frictionless legitimate payments.  
**Risks:** False positives may cause financial exclusion (especially for minorities or unusual devices), disparate impact from biased signals (domains, devices, geographies), chilling effect from fingerprinting, and lack of transparency in decisions.

In summary, this academic work carries minimal risk but demonstrates why real fraud-detection systems require strong ethical, privacy-by-design, and fairness frameworks.

---
## Requirements

manually in terminal:
```bash
  pip install \
  pandas>=2.0 \
  numpy>=1.24 \
  seaborn>=0.12 \
  matplotlib>=3.7 \
  lightgbm>=4.0 \
  xgboost>=2.0 \
  catboost>=1.2 \
  optuna>=3.0 \
  scikit-learn>=1.3 \
  joblib>=1.2 \
  altair>=5.0
```

---
## How to Reproduce
### **1. Clone the repository**
```bash
git clone https://github.com/vdafeider/ieee_fraud_detection_analysis_and_ml.git
cd ieee_fraud_detection_analysis_and_ml
```
### **2. Install dependencies**
See above [requirements](#Requirements)

### **3. Run the notebook**
```bash
jupyter notebook
```
Open `Transaction_Fraud_Detection_ETA_ML.ipynb` → **Restart & Run All**.

### **4. Generate test predictions**
Notebook already contains final prediction code using saved bundle
Output: columns isFraud_float (probability) + isFraud (binary @ threshold ≈0.27)

###
**To open Power Bi dashboard in app:**
1. Install **Power BI Desktop** (version 2.149.1429.0 recommended) from the [official website](https://www.microsoft.com/en-us/download/details.aspx?id=58494).  
2. Open the file: `dataset_overview.pbix`  
3. To update the dataset, reconnect the train `.csv` files from the Kaggle using **Power Query**.

---
## Contributing
Feel free to fork and submit PRs.
Especially welcome:
* Additional strong features (time-based, card grouping, interaction terms)
* Alternative models (TabNet, NN, stacking) or Deep Learning
* Better handling of high-cardinality categoricals
* Create a dynamic threshold that changes depending on the transaction amount to minimize losses

Please follow best practices:
- Fork the repository
- Create a feature branch
- Commit with clear messages
- Open a Pull Request

If you plan to contribute regularly, consider adding:
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`

---

## License
This project is licensed under the **MIT License**.

---

## Credits and Acknowledgements
The content of this project represents the understanding gained from the walkthrough projects provided by **Code Institute**.  

Issues encountered during development were resolved by **leveraging official documentation, community forums, and best practices** from resources including Stack Overflow, Python library documentation, and YouTube tutorials. 

Great thanks to Dataset owners for data share: Vesta Corporation via IEEE-CIS Fraud Detection @ Kaggle

A huge thanks to **John Anih**, who introduced me to this course.

---

## Author
**Volodymyr Babunych**  
📧 vbabunych@gmail.com  
📍 United Kingdom  
🗓️ January 23, 2026

