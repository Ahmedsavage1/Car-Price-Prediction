# Car Price Prediction

Predicting used car prices from listing data (7,253 cars) using a leakage-safe, production-ready ML pipeline.

## Project Structure

```
Car Price Prediction/
├── data/
│   ├── raw/                          # original dataset
│   └── processed/                    # cleaned data
├── models/                           # model + preprocessing files
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # basic cleaning
│   ├── 02_data_preprocessing.ipynb   # split, impute, encode, scale
│   ├── 03_eda.ipynb                  # EDA
│   └── 04_modeling.ipynb             # model training + tuning
├── app.py                            # Streamlit app
├── requirements.txt                  # dependencies
└── README.md
```

## Dataset

This project uses the [Used Cars Dataset](https://www.kaggle.com/datasets/ayushparwal2026/cars-dataset) from Kaggle (not included in this repo — see below).

To run the notebooks, download `train-data.csv`, rename it to `used_cars_data.csv`, and place it in `data/raw/`.

## Pipeline

**1. Data Cleaning (row-level, before split — no leakage possible)**
- Dropped `S.No.` (useless index) and `New_Price` (86% missing)
- Dropped rows with missing target (`Price`)
- Split embedded units from `Mileage`, `Engine`, `Power`; recovered a hidden "null bhp" text value
- Dropped Electric cars (Mileage doesn't apply to them)
- Engineered `Car_Age` (from Year) and `Brand` (from Name), fixed naming inconsistencies

**2. Preprocessing (train/test split first, everything else fit on train only)**
- 80/20 train/test split
- Imputed `Engine`/`Power` (median by Brand) and `Seats` (mode by Brand), with overall fallback — statistics computed from X_train only
- Removed true duplicate rows (identical features **and** price)
- Grouped rare brands (<100 occurrences in X_train) into "Other"
- One-hot encoded categorical columns; X_test columns aligned to X_train's
- Ordinal encoded `Owner_Type`
- Scaled numeric columns with MinMaxScaler, fit on X_train only

**3. Modeling**
- Tried Linear Regression (baseline), Decision Tree, Random Forest, XGBoost
- Tuned each with GridSearchCV (5-fold CV), including a second, more heavily regularized search to test overfitting reduction
- Compared every run on Train R², Test R², RMSE, and the Train–Test gap (overfitting signal)

**4. Deployment**
- Best model, scaler, rare-brand list, and final feature-column order all saved as `.pkl` artifacts
- Served through a Streamlit app that replicates the exact preprocessing pipeline on new user input before prediction

## Results

| Model | Test R² | Test RMSE | Train–Test Gap |
|---|---|---|---|
| Linear Regression | 0.729 | 5.64 | 0.008 |
| Decision Tree (tuned) | 0.773 | 5.16 | 0.165 |
| Random Forest (tuned) | 0.875 | 3.84 | 0.111 |
| **XGBoost (tuned) — final model** | **0.878** | **3.78** | 0.108 |

A second round of heavier regularization (Random Forest, XGBoost) reduced the overfitting gap slightly but did **not** improve Test R² — performance plateaued around 0.87–0.88, suggesting the ceiling is set more by available features than by model tuning.

## Known Limitations

- **Performance ceiling ~0.88 R²**: further hyperparameter tuning stopped helping; likely needs better features (e.g. exact model trim, condition, accident history — none of which exist in this dataset) rather than a different model or more tuning.
- **Rare brands lose granularity**: brands under 100 occurrences in training data are grouped into "Other," so the model can't distinguish, e.g., Lamborghini from Bentley.
- **No temporal/location pricing signal**: the model doesn't account for regional demand shifts or time-of-sale market trends beyond what's implicitly in `Car_Age`.
- **Trained on a fixed historical snapshot**: prices reflect the dataset's collection period; may drift from current market prices over time and would need periodic retraining.
- **One duplicate-handling judgment call**: rows with identical features but different prices were deliberately kept (treated as real market variation, not data errors) — this is a modeling assumption, not a guaranteed fact about the data.