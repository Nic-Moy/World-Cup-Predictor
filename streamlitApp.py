import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Match Predictor",
    page_icon="⚽",
    layout="wide",
)

CLASS_NAMES = ['Home win', 'Draw', 'Away win']
WINDOW = 5  # rolling form window

MAJOR = ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'African Cup of Nations',
         'AFC Asian Cup', 'UEFA Nations League', 'CONCACAF Nations League']

DATA_URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'


# ──────────────────────────────────────────────────────────────────────────
# Data loading + feature engineering (cached so it only runs once)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading match history...")
def load_raw_data():
    raw = pd.read_csv(DATA_URL, parse_dates=['date'])
    df = raw.dropna(subset=['home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df = df[df['date'] >= '2000-01-01'].sort_values('date').reset_index(drop=True)
    return df


def build_team_log(matches):
    sides = {
        'home': ('home_team', 'home_score', 'away_score'),
        'away': ('away_team', 'away_score', 'home_score'),
    }
    frames = []
    for side, (tcol, gf, ga) in sides.items():
        f = matches[['match_id', 'date', tcol, gf, ga]].copy()
        f.columns = ['match_id', 'date', 'team', 'goals_for', 'goals_against']
        f['side'] = side
        frames.append(f)
    log = pd.concat(frames).sort_values('date').reset_index(drop=True)
    log['gd'] = log['goals_for'] - log['goals_against']
    log['pts'] = np.select([log['gd'] > 0, log['gd'] == 0], [3, 1], default=0)
    return log


def importance(t):
    if t == 'Friendly':
        return 0
    if any(t == m for m in MAJOR):
        return 2
    return 1


FEATURES = [
    'form_pts_diff', 'form_gd_diff', 'exp_diff',
    'home_form_pts', 'home_form_gd', 'away_form_pts', 'away_form_gd',
    'is_home_advantage', 'is_neutral', 'tournament_importance',
]


@st.cache_data(show_spinner="Engineering features...")
def engineer_features(df):
    df = df.copy()
    conditions = [df['home_score'] > df['away_score'], df['home_score'] == df['away_score']]
    df['result'] = np.select(conditions, [0, 1], default=2)
    df['match_id'] = df.index

    team_log = build_team_log(df)
    g = team_log.groupby('team')
    team_log['form_pts'] = g['pts'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
    team_log['form_gd'] = g['gd'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
    team_log['matches_played'] = g.cumcount()

    home_feats = (team_log[team_log['side'] == 'home']
                  .set_index('match_id')[['form_pts', 'form_gd', 'matches_played']]
                  .rename(columns=lambda c: 'home_' + c))
    away_feats = (team_log[team_log['side'] == 'away']
                  .set_index('match_id')[['form_pts', 'form_gd', 'matches_played']]
                  .rename(columns=lambda c: 'away_' + c))

    data = df.set_index('match_id').join(home_feats).join(away_feats)

    data['form_pts_diff'] = data['home_form_pts'] - data['away_form_pts']
    data['form_gd_diff'] = data['home_form_gd'] - data['away_form_gd']
    data['exp_diff'] = data['home_matches_played'] - data['away_matches_played']
    data['is_home_advantage'] = (~data['neutral']).astype(int)
    data['is_neutral'] = data['neutral'].astype(int)
    data['tournament_importance'] = data['tournament'].apply(importance)

    model_df = data.dropna(subset=['home_form_pts', 'away_form_pts']).copy()
    model_df = model_df.sort_values('date')

    return model_df, team_log


# ──────────────────────────────────────────────────────────────────────────
# Model training (cached as a resource so it persists across reruns)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training XGBoost model...")
def train_model(model_df):
    X = model_df[FEATURES]
    y = model_df['result']

    split_idx = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()

    model = XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='mlogloss',
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    metrics = {
        'baseline_acc': baseline_acc,
        'acc': acc,
        'macro_f1': macro_f1,
        'cm': cm,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'split_date': model_df['date'].iloc[split_idx].date(),
    }
    return model, metrics


# ──────────────────────────────────────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────────────────────────────────────
def latest_form(team_log, team):
    rows = team_log[team_log['team'] == team].dropna(subset=['form_pts'])
    if rows.empty:
        return None
    last = rows.sort_values('date').iloc[-1]
    return last['form_pts'], last['form_gd'], last['matches_played']


def predict_match(model, team_log, home, away, neutral, tourn_importance):
    h = latest_form(team_log, home)
    a = latest_form(team_log, away)
    if h is None or a is None:
        return None

    h_pts, h_gd, h_exp = h
    a_pts, a_gd, a_exp = a

    row = pd.DataFrame([{
        'form_pts_diff': h_pts - a_pts,
        'form_gd_diff': h_gd - a_gd,
        'exp_diff': h_exp - a_exp,
        'home_form_pts': h_pts, 'home_form_gd': h_gd,
        'away_form_pts': a_pts, 'away_form_gd': a_gd,
        'is_home_advantage': 0 if neutral else 1,
        'is_neutral': 1 if neutral else 0,
        'tournament_importance': tourn_importance,
    }])[FEATURES]

    proba = model.predict_proba(row)[0]
    return proba


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.title("⚽ World Cup Match Predictor")
st.caption(
    "XGBoost model trained on international match results since 2000. "
    "Pick two teams and a match context to see predicted outcome probabilities."
)

raw_df = load_raw_data()
model_df, team_log = engineer_features(raw_df)
model, metrics = train_model(model_df)

teams = sorted(team_log['team'].unique())

tab_predict, tab_model = st.tabs(["🔮 Predict a Matchup", "📊 Model Performance"])

# ── Predict tab ─────────────────────────────────────────────────────────
with tab_predict:
    col1, col2, col3 = st.columns([2, 2, 1.4])

    with col1:
        home_team = st.selectbox("Home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)

    with col2:
        away_options = [t for t in teams if t != home_team]
        default_away = "Argentina" if "Argentina" in away_options else away_options[0]
        away_team = st.selectbox("Away team", away_options, index=away_options.index(default_away))

    with col3:
        neutral = st.checkbox("Neutral venue", value=True, help="Check this for World Cup-style matches at a neutral ground.")
        tourn_label = st.selectbox(
            "Match context",
            ["Friendly", "Qualifier / Regional Cup", "Major Tournament (World Cup, Euro, etc.)"],
            index=2,
        )
        tourn_map = {"Friendly": 0, "Qualifier / Regional Cup": 1, "Major Tournament (World Cup, Euro, etc.)": 2}
        tourn_importance = tourn_map[tourn_label]

    st.divider()

    if st.button("Predict result", type="primary", use_container_width=False):
        proba = predict_match(model, team_log, home_team, away_team, neutral, tourn_importance)

        if proba is None:
            st.error("No form history found for one of these teams — try a different matchup.")
        else:
            labels = [f"{home_team} win", "Draw", f"{away_team} win"]
            colors = ["#27AE60", "#999999", "#E74C3C"]

            pick_idx = int(np.argmax(proba))
            st.success(f"🏆 Most likely outcome: **{labels[pick_idx]}** ({proba[pick_idx]*100:.1f}%)")

            cols = st.columns(3)
            for c, label, p, color in zip(cols, labels, proba, colors):
                with c:
                    st.metric(label, f"{p*100:.1f}%")

            fig = go.Figure(go.Bar(
                x=labels, y=proba * 100, marker_color=colors, text=[f"{p*100:.1f}%" for p in proba],
                textposition='outside',
            ))
            fig.update_layout(
                yaxis_title="Probability (%)", yaxis_range=[0, 100],
                showlegend=False, height=380,
                title=f"{home_team} vs {away_team} — {'Neutral venue' if neutral else f'{home_team} hosting'}",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick two teams above and click **Predict result**.")

# ── Model performance tab ──────────────────────────────────────────────
with tab_model:
    st.subheader("Holdout test performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['acc']*100:.1f}%")
    c2.metric("Baseline accuracy", f"{metrics['baseline_acc']*100:.1f}%")
    c3.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    c4.metric("Lift over baseline", f"{(metrics['acc']-metrics['baseline_acc'])*100:+.1f} pp")

    st.caption(
        f"Trained on {metrics['n_train']:,} matches up to {metrics['split_date']}, "
        f"tested on {metrics['n_test']:,} matches after that date."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion matrix (row-normalized)")
        fig_cm = px.imshow(
            metrics['cm'], x=CLASS_NAMES, y=CLASS_NAMES, color_continuous_scale="Blues",
            text_auto=".2f", zmin=0, zmax=1,
            labels=dict(x="Predicted", y="Actual", color="Proportion"),
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Feature importance")
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig_imp = px.bar(
            x=importances.values, y=importances.index, orientation='h',
            labels={'x': 'Importance (gain)', 'y': ''},
        )
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.caption("Label scheme: 0 = home win, 1 = draw, 2 = away win.")