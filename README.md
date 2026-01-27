# ML Assignment 2 – Classification Models and Streamlit App

## Problem statement
Build and compare six classification models on a public dataset, report standard metrics, and expose an interactive Streamlit UI for model exploration and predictions. Deploy the app on Streamlit Community Cloud and provide links plus a BITS Lab execution screenshot.

## Dataset description
- Dataset: Bank Marketing (UCI) – direct marketing campaign outcome (binary target `y`: yes/no).
- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv
- Instances: 41,188; Features: 20 input features (categorical + numerical); Target: 1 binary column.
- Rationale: Public, license-free for research/education; satisfies >=12 features and >=500 rows requirement.

## Models and metrics
Trained on an 80/20 stratified split with preprocessing (one-hot for categoricals, scaling for numerics). Metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1  | MCC |
|----------------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression  | TODO     | TODO| TODO      | TODO   | TODO| TODO|
| Decision Tree        | TODO     | TODO| TODO      | TODO   | TODO| TODO|
| kNN                  | TODO     | TODO| TODO      | TODO   | TODO| TODO|
| Naive Bayes          | TODO     | TODO| TODO      | TODO   | TODO| TODO|
| Random Forest (Ens.) | TODO     | TODO| TODO      | TODO   | TODO| TODO|
| XGBoost (Ens.)       | TODO     | TODO| TODO      | TODO   | TODO| TODO|

### Observations
| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | TODO |
| Decision Tree        | TODO |
| kNN                  | TODO |
| Naive Bayes          | TODO |
| Random Forest (Ens.) | TODO |
| XGBoost (Ens.)       | TODO |

## Project structure
```
project-folder/
│-- app.py
│-- train_models.py
│-- requirements.txt
│-- README.md
│-- model/        # saved pipelines (.joblib) and metrics (.csv/.json)
```

## How to run locally
1. Install deps: `pip install -r requirements.txt`
2. Train and save models: `python train_models.py`
3. Launch app: `streamlit run app.py`

## Deployment (Streamlit Community Cloud)
- Connect GitHub repo, select main branch and `app.py`, deploy. Ensure `requirements.txt` is present. The app reads saved pipelines from `model/` and downloads the dataset if needed.

## Notes for submission
- Include the GitHub repo link, live Streamlit app link, README content (this file) in the submission PDF.
- Capture and attach one screenshot of the app running on BITS Virtual Lab.
