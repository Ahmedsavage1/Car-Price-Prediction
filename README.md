# Car Price Prediction

Predicting used car prices from listing data (7,253 cars) using a leakage-safe, production-ready ML pipeline, deployed as a Streamlit app.

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

This project uses the [Used Cars Dataset](https://www.kaggle.com/datasets/ayushparwal2026/cars-dataset) from Kaggle (not included in this repo — see below). Prices are in **Lakhs INR** (1 Lakh = 100,000 Indian Rupees).

To run the notebooks, download the dataset, rename the CSV to `used_cars_data.csv`, and place it in `data/raw/`.

## Pipeline

- **Cleaning**: dropped unreliable/redundant columns, fixed unit-embedded numeric fields, engineered `Car_Age` and `Brand`, removed one impossible outlier (a car with ~6.5M km driven)
- **Preprocessing**: train/test split first, then imputation, encoding, and scaling — all fit on the training set only, to avoid data leakage
- **Modeling**: compared Linear Regression, Decision Tree, Random Forest, and XGBoost, tuned with GridSearchCV (cross-validated, not a single train/test split)
- **Deployment**: best model and all preprocessing artifacts saved and served through a Streamlit app, which replicates the exact same preprocessing on new user input before predicting

## Results

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear Regression | 0.729 | 5.64 |
| Decision Tree (tuned) | 0.727 | 5.46 |
| Random Forest (tuned) | 0.879 | 3.64 |
| **XGBoost (tuned) — final model** | **0.912** | **3.11** |

Removing one bad data point (an impossible mileage value) improved every model, especially XGBoost.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app shows every real brand name in the dropdown. Rare brands (uncommon in training data) are mapped to "Other" internally before prediction — the user doesn't need to know or care about this.

## Known Limitations

- Rare brands lose individual identity (grouped into "Other")
- No signal for regional demand shifts or time-of-sale market trends
- Trained on a historical snapshot — would need retraining to stay accurate over time