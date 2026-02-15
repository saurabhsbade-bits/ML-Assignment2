# ML Assignment 2 – Classification Models and Streamlit App

## Problem statement
Build and compare six classification models on a public dataset, report standard metrics, and deploy an interactive Streamlit web application for model exploration and predictions.

## Dataset description
- **Dataset**: Adult Census Income (UCI/OpenML) – binary classification to predict if income >50K
- **Source**: https://www.openml.org/d/1590 (fetched via `fetch_openml('adult', version=2)`)
- **Size**: ~48,842 instances with 14 input features (mix of categorical and numerical)
- **Target**: Binary income classification (<=50K or >50K)

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
│-- app.py                    # Streamlit web application
│-- requirements.txt          # Python dependencies
│-- README.md                 # Project documentation
│-- data/                     # Sample test CSV files
│-- model/
    │-- train_models.py       # Training script for all models
    │-- *.joblib              # Saved model pipelines
    │-- metrics.csv/json      # Model evaluation results
```

## How to run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Train and save models: `python model/train_models.py`
3. Launch app: `streamlit run app.py`

## Streamlit App Features
- **Model selection**: Choose from six trained classification models via sidebar dropdown
- **Hold-out test performance**: View evaluation metrics and confusion matrix on 20% hold-out test split
- **Dataset upload**: Upload CSV files via sidebar for batch predictions on custom test data
- **Metrics display**: Comprehensive metrics (accuracy, AUC, precision, recall, F1, MCC) with confusion matrix
- **Classification report**: Detailed per-class metrics for uploaded data (when labels are provided)
- **Downloadable predictions**: Export predictions as CSV file

## Deployment
Deployed on Streamlit Community Cloud. The app automatically loads pre-trained model pipelines from the `model/` directory.

## Links
- **GitHub repository**: https://github.com/saurabhsbade-bits/ML-Assignment2
- **Live Streamlit app**: https://2025aa05203-ml.streamlit.app/
