
# RainLens — Sri Lankan Rainfall Prediction

A machine learning project comparing Bagging, Boosting and Stacking ensemble learning techniques for rainfall prediction using historical Sri Lankan weather data.

## Project Overview

- Dataset: 25,000 Sri Lankan weather observations
- Input features: 49
- Target: Next-day rainfall
- Dashboard: Streamlit

## Machine Learning Models

1. Decision Tree (Baseline)
2. Bagging Classifier
3. Random Forest
4. AdaBoost
5. XGBoost
6. Stacking Classifier

## Results

The Stacking Classifier achieved the highest F1-score of 85.23% and accuracy of 80.36% in the historical evaluation.

Random Forest achieved the highest ROC-AUC of 87.96%.

## Dashboard Features

- Interactive model performance comparison
- Historical rainfall prediction
- Experimental future rainfall prediction using Open-Meteo
- Dataset preview and download

## Running the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

## Project Files

- `app.py` — Streamlit dashboard
- `notebooks/ensemble_learning.ipynb` — Machine learning experiments
- `data/` — Weather datasets
- `models/` — Trained models and saved results
- `.streamlit/` — Dashboard configuration




