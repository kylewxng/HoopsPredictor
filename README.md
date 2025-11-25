# HoopsPredictor — NBA Prop Prediction App

HoopsPredictor is a machine learning-powered web application that predicts NBA player Points, Rebounds, and Assists for upcoming games.  
The system uses XGBoost ensemble models trained on engineered performance features and delivers predictions in real-time through a Streamlit-based user interface.  
Users simply select a player, opponent, and whether the game is home or away.

---

## Live Web App

Access the Streamlit deployed model here:  
https://hoopspredictor.streamlit.app/  

---

## Features

- Predicts individual performance for Points, Rebounds, and Assists
- Built on 16+ engineered predictive features:
  - Rolling averages for PTS, REB, AST
  - Historical matchup performance
  - Rest days between games
  - Home vs. Away splits
  - Opponent defensive difficulty metrics
- Time-series aware validation prevents information leakage
- Streamlit UI enables accessible projections without manual stat entry

---

## Model Performance

| Category | MAE | RMSE |
|----------|-----|------|
| Points   | ~4.6 | ~6.0 |
| Rebounds | ~1.9 | ~2.6 |
| Assists  | ~1.3 | ~1.8 |

Performance represents improvement of approximately 20–35% over season-average baseline predictions.

---

## Methodology

**Dataset**: Over 1,600 games from the 2024–2025 NBA season  
**Modeling**: XGBoost Regressors per stat category with GridSearchCV hyperparameter tuning  
**Deployment**: Streamlit Cloud serving serialized models (pickle)

---

## Tech Stack

| Area | Tools |
|------|------|
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Development | Google Colab, Python |
| Deployment | Streamlit Cloud |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git + GitHub |

---

## Repository Structure

```text
HoopsPredictor/
 ├── models/              # Serialized XGBoost models
 ├── notebooks/           # Training and analysis notebooks
 ├── data/                # Data references / preprocessing outputs
 ├── streamlit_app.py     # App inference + UI
 ├── requirements.txt     # Dependencies
 └── README.md            # Documentation

