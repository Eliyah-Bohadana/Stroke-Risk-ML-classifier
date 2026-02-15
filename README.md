# 🧠 Stroke Risk Prediction - Data Exploration & Preprocessing

## 📌 Project Overview
This project prepares a healthcare dataset for a machine learning model that will predict stroke risk using demographic, medical, and lifestyle factors. The goal of this stage is to build a clean, reproducible data pipeline: understand the dataset, fix data quality issues, transform all features into numeric form, and export a model-ready dataset.

---

## 📊 Dataset
**Source:** Kaggle Stroke Prediction Dataset  
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

- **Rows:** 5110  
- **Columns:** 12  
- **Target:** `stroke` (0 = no stroke, 1 = stroke)

**Original columns:**
- `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`,
  `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`

---

## 🔎 1) Exploratory Data Analysis (EDA)
**Notebook:** `notebooks/01_exploration.ipynb`

### What we checked
1. **Dataset size & structure**
   - Verified dataset loads correctly
   - Confirmed shape (rows/columns) and feature names

2. **Data types**
   - Identified numeric features vs categorical (text) features using `df.info()`

3. **Missing values**
   - Used `df.isna().sum()` to locate missing data
   - Found **~201 missing values in `bmi`** (~4% of dataset)
   - All other columns had **0 missing values**

4. **Target distribution (class imbalance)**
   - Checked `df["stroke"].value_counts()` / `normalize=True`
   - Confirmed `stroke` is **highly imbalanced** (stroke cases are rare)
   - This informs future modeling (accuracy alone is misleading)

5. **Categorical value distributions**
   - Used `value_counts()` to inspect categories and detect inconsistencies
   - Identified categorical columns:
     - `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
   - Found key issues:
     - **Rare category:** `Never_worked` in `work_type` (22 rows, <0.5%)
     - **Pseudo-category:** `Unknown` in `smoking_status`

6. **Numeric sanity checks**
   - Used `df.describe()` to confirm numeric ranges were reasonable
   - No obvious invalid values were found

---

## 🧹 2) Data Cleaning & Preprocessing
**Notebook:** `notebooks/02_preprocessing.ipynb`  
**Goal:** produce a fully numeric, model-ready dataset with no missing values.

### Step 1 - Drop non-predictive identifier
- Dropped `id` because it is an identifier and not predictive (prevents leakage-like behavior).

### Step 2 — Handle missing values (BMI)
- Imputed missing `bmi` values using the **median**:
  - Median is robust to outliers
  - Keeps all rows (avoids dropping data)

### Step 3 - Encode binary categorical features (0/1)
Converted binary text fields into numeric:
- `gender`: Male = 1, Female = 0  
- `Residence_type`: Urban = 1, Rural = 0  
- `ever_married`: Yes = 1, No = 0  

Reason: binary variables don’t need one-hot encoding; 0/1 keeps features simple and interpretable.

### Step 4 - Merge rare category (stability improvement)
- `work_type` had an extremely rare category:
  - `Never_worked` = 22 rows
- We merged `Never_worked` into a more stable group (`children`) to reduce feature sparsity and noise.

### Step 5 - One-hot encode multi-class categorical features
Some categorical columns have **more than 2 categories** and no natural order, so we one-hot encoded them:
- `smoking_status`
- `work_type`

We used `pd.get_dummies(..., drop_first=True)`:
- Preserves meaning without creating fake numeric ordering
- `drop_first=True` reduces multicollinearity / dummy-variable trap

### Step 6 - Final validation checks (quality gates)
Before saving, we verified:
- **No missing values remain**
  - `df.isna().sum().sum() == 0`
- **No text/object columns remain**
  - `df.select_dtypes("object").columns` is empty
- **Target column is preserved**
  - `stroke` remains as 0/1

### Step 7 - Export processed dataset
Saved the final model-ready dataset to:
- `data/processed/stroke_clean.csv`

This file is:
- fully numeric
- contains no missing values
- ready for train/test splitting and model training

---

## 📁 Project Structure (Current)
