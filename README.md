# ⚽ World Cup Match Predictor

Predict the outcome of any international football matchup — **home win, draw, or away win** — with an XGBoost model trained on every international result since 2000.

Pick two teams, set the match context, and read the predicted probabilities in a clean broadcast-style UI.

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-CCFF00)

---

## What it does

- **Predict a matchup** — choose home/away teams, neutral venue, and match importance (friendly, qualifier, or major tournament). Get win/draw/loss probabilities.
- **Model performance** — holdout test accuracy, macro-F1, lift over baseline, a confusion matrix, and feature importances.

## How it works

| Step | Detail |
|------|--------|
| **Data** | [martj42/international_results](https://github.com/martj42/international_results) — every international match, filtered to 2000+. |
| **Features** | Rolling 5-match form (points + goal difference), form differentials, experience gap, home advantage, neutral venue, tournament importance. |
| **Model** | `XGBoostClassifier` — multiclass softprob, 300 trees, depth 4, learning rate 0.05. |
| **Validation** | Time-based 80/20 split (train on older matches, test on newer) to avoid leakage. |

All data loading, feature engineering, and model training are cached, so the model trains **once per server** and is shared across all visitors.

## Run locally

```bash
# 1. clone
git clone https://github.com/Nic-Moy/World-Cup-Predictor.git
cd World-Cup-Predictor

# 2. install
pip install -r requirements.txt

# 3. launch
streamlit run streamlitApp.py
```

App opens at `http://localhost:8501`. Match history downloads automatically on first run.

## Tech stack

`Python` · `Streamlit` · `XGBoost` · `scikit-learn` · `pandas` · `Plotly`

## Notes

- Predictions reflect each team's **most recent form** in the dataset, not live squad/injury data.
- A draw is genuinely common in football, so a high draw probability is expected for evenly matched sides.
