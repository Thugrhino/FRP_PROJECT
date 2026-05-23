# FRP Project

Professional multimorbidity risk prediction project built from the provided datasets.

## What Is Improved

- Clean project structure with raw data, processed data, models, plots, reports, source code, and tests.
- Dataset cleaning for invalid age, BMI, clinical ranges, duplicate rows, and invalid categorical values.
- Fresh feature engineering per disease.
- Cross-validated model comparison across Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
- Suspicious-perfect-model detection, so models that hit near-100% performance are flagged and skipped when a stable non-perfect model exists.
- White and green Streamlit medical dashboard with transparent panels and high-contrast text.

## Run Training

```bash
python train_models.py
```

## Run Checks

```bash
python tests/run_checks.py
```

## Run Dashboard

```bash
streamlit run streamlit_app.py
```

## Outputs

- `data/processed/*_cleaned.csv`
- `data/processed/*_engineered.csv`
- `models/*_bundle.pkl`
- `reports/metrics_*.csv`
- `reports/experiment_report.txt`
- `plots/*.png`

## Important Note

The hypertension file is a heart-disease-style dataset with a target named `target`.
The project keeps the user-facing name "Hypertension", but reports clearly mark it as a hypertension-risk proxy dataset rather than a direct clinical hypertension diagnosis.
