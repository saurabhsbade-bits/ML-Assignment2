# ML Assignment 2 – Classification Models and Streamlit App

## Problem statement
Build and compare six classification models on a public dataset, report standard metrics, and expose an interactive Streamlit UI for model exploration and predictions. Deploy the app on Streamlit Community Cloud and provide links plus a BITS Lab execution screenshot.

## Dataset description
- Dataset: Adult Census Income (UCI/OpenML) – predict if income >50K.
- Source: https://www.openml.org/d/1590 (fetched via `fetch_openml('adult', version=2)`)
- Instances: ~48,842; Features: 14 input features (categorical + numerical); Target: binary income column.
- Rationale: Public, research-friendly license; meets >=12 features and >=500 rows requirement.

## Models and metrics
Trained on an 80/20 stratified split with preprocessing (one-hot for categoricals, scaling for numerics). Metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1  | MCC |
|----------------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression  | 0.8051   | 0.8257| 0.6974    | 0.3773 | 0.4897| 0.4098|
| Decision Tree        | 0.7725   | 0.6971| 0.5406    | 0.5464 | 0.5435| 0.3920|
| kNN                  | 0.8144   | 0.8217| 0.6998    | 0.4398 | 0.5401| 0.4496|
| Naive Bayes          | 0.7881   | 0.8245| 0.6558    | 0.3051 | 0.4164| 0.3411|
| Random Forest (Ens.) | 0.8024   | 0.8190| 0.6237    | 0.5116 | 0.5621| 0.4397|
| XGBoost (Ens.)       | 0.8309   | 0.8614| 0.8139    | 0.4117 | 0.5468| 0.4964|

### Observations
| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Linear classifier with good AUC (0.8257) but moderate F1 (0.4897); handles class imbalance reasonably. |
| Decision Tree        | Balanced recall/precision (0.5464/0.5406, F1=0.5435) but lower AUC (0.6971); tends to overfit on majority class. |
| kNN                  | Strong accuracy (0.8144) and AUC (0.8217); good at capturing local patterns; slower inference for large data. |
| Naive Bayes          | High AUC (0.8245) but lowest recall (0.3051); assumes feature independence; biased toward majority class. |
| Random Forest (Ens.) | Balanced F1 (0.5621) and moderate AUC (0.8190); robust ensemble, less prone to overfitting than single tree. |
| XGBoost (Ens.)       | Best overall: highest accuracy (0.8309), AUC (0.8614), precision (0.8139), and MCC (0.4964); strong gradient boosting. |

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
