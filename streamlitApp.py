import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import world_cup_predictor as wc  # shared model + feature pipeline (single source of truth)
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nic's World Cup Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────
# Design system — "night-match broadcast" aesthetic
#   deep pitch background · chalk lines · electric-lime accent ·
#   condensed sports typography (Anton / Oswald) · tabular stat type
# ──────────────────────────────────────────────────────────────────────────
LIME = "#CCFF00"
HOME = "#19E3A0"   # vivid teal-green
DRAW = "#8A99A0"   # slate
AWAY = "#FF5B47"   # vivid red
INK = "#0A1410"
PANEL = "#10211A"
MUTED = "#7E9088"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@400;500;600;700&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=Space+Mono:wght@400;700&display=swap');
      /* ----- base canvas: pitch at night ----- */
      .stApp {
        background:
          radial-gradient(1200px 600px at 12% -10%, rgba(204,255,0,0.07), transparent 60%),
          radial-gradient(1000px 700px at 100% 0%, rgba(25,227,160,0.06), transparent 55%),
          repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 120px),
          linear-gradient(180deg, #0A1410 0%, #081410 100%);
        color: #E8F0EC;
      }
      /* subtle film grain */
      .stApp::before {
        content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
        opacity: 0.035;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
      }
      .block-container { padding-top: 2.2rem; max-width: 1180px; position: relative; z-index: 1; }

      /* ----- hide default streamlit chrome ----- */
      #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

      /* ----- typography ----- */
      html, body, [class*="css"], .stMarkdown, p, label, span, div {
        font-family: 'DM Sans', system-ui, sans-serif;
      }

      /* ----- hero banner ----- */
      .wc-hero {
        position: relative; margin: 0 0 1.6rem 0; padding: 2.2rem 2.4rem 2rem;
        border: 1px solid rgba(204,255,0,0.18);
        border-radius: 18px; overflow: hidden;
        background:
          linear-gradient(120deg, rgba(204,255,0,0.10), transparent 42%),
          radial-gradient(600px 200px at 90% 120%, rgba(25,227,160,0.12), transparent),
          #0C1B15;
      }
      .wc-hero::after {  /* center-circle pitch motif */
        content: ""; position: absolute; right: -90px; top: 50%; transform: translateY(-50%);
        width: 260px; height: 260px; border-radius: 50%;
        border: 2px solid rgba(255,255,255,0.06);
        box-shadow: 0 0 0 1px rgba(204,255,0,0.05);
      }
      .wc-kicker {
        font-family: 'Oswald', sans-serif; font-weight: 600; letter-spacing: 0.42em;
        text-transform: uppercase; font-size: 0.72rem; color: #CCFF00; margin: 0 0 0.5rem;
      }
      .wc-title {
        font-family: 'Anton', sans-serif; font-weight: 400; letter-spacing: 0.01em;
        text-transform: uppercase; line-height: 0.92;
        font-size: clamp(2.6rem, 6vw, 4.4rem); margin: 0; color: #F4FBF6;
        text-shadow: 0 2px 30px rgba(0,0,0,0.4);
      }
      .wc-title em { font-style: normal; color: #CCFF00; }
      .wc-sub { color: #9FB2A8; font-size: 0.98rem; max-width: 640px; margin: 0.85rem 0 0; line-height: 1.5; }

      /* ----- tabs as broadcast nav ----- */
      .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.07); }
      .stTabs [data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.12em;
        font-weight: 600; font-size: 0.82rem; color: #7E9088; padding: 0.6rem 0.2rem;
      }
      .stTabs [aria-selected="true"] { color: #F4FBF6 !important; }
      .stTabs [data-baseweb="tab-highlight"] { background: #CCFF00; height: 3px; }

      /* ----- section labels ----- */
      .stSelectbox label, .stCheckbox label p {
        font-family: 'Oswald', sans-serif !important; text-transform: uppercase;
        letter-spacing: 0.14em; font-size: 0.72rem !important; color: #8FA399 !important; font-weight: 600;
      }
      /* selectbox / input shells */
      .stSelectbox [data-baseweb="select"] > div {
        background: #0E1D17; border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
        transition: border-color .18s ease, box-shadow .18s ease;
      }
      .stSelectbox [data-baseweb="select"] > div:hover { border-color: rgba(204,255,0,0.4); }
      .stSelectbox [data-baseweb="select"] > div:focus-within {
        border-color: #CCFF00; box-shadow: 0 0 0 3px rgba(204,255,0,0.14);
      }

      /* ----- primary button ----- */
      .stButton > button {
        font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.16em;
        font-weight: 700; font-size: 0.92rem; color: #08140F !important;
        background: #CCFF00; border: 0; border-radius: 12px; padding: 0.7rem 2.2rem;
        box-shadow: 0 8px 24px rgba(204,255,0,0.22); transition: transform .15s ease, box-shadow .15s ease;
      }
      .stButton > button:hover {
        transform: translateY(-2px); background: #D8FF33;
        box-shadow: 0 12px 32px rgba(204,255,0,0.34); color: #08140F !important;
      }
      .stButton > button:active { transform: translateY(0); }

      /* ----- metric cards (model tab) ----- */
      [data-testid="stMetric"] {
        background: #0E1D17; border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 1rem 1.2rem;
      }
      [data-testid="stMetricLabel"] p {
        font-family: 'Oswald', sans-serif !important; text-transform: uppercase;
        letter-spacing: 0.12em; font-size: 0.7rem !important; color: #8FA399 !important;
      }
      [data-testid="stMetricValue"] { font-family: 'Anton', sans-serif; color: #F4FBF6; }

      /* ----- scoreboard result ----- */
      .wc-board {
        display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 1rem;
        padding: 1.8rem 2rem; border-radius: 18px; margin: 0.4rem 0 1.4rem;
        background: linear-gradient(180deg, #0E1D17, #0B1813);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .wc-side { text-align: center; }
      .wc-side .name {
        font-family: 'Oswald', sans-serif; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; font-size: 1rem; color: #C7D6CE; margin-bottom: 0.3rem;
      }
      .wc-side .pct { font-family: 'Anton', sans-serif; font-size: 3.6rem; line-height: 1; }
      .wc-vs {
        font-family: 'Oswald', sans-serif; font-weight: 700; color: #5E6F67;
        text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.8rem;
        border-left: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.08);
        padding: 0 1.4rem;
      }
      .wc-vs .draw-pct { display:block; font-family:'Anton',sans-serif; font-size:1.5rem; color:#B7C4BC; margin-top:0.2rem; }

      /* possession-style stacked probability bar */
      .wc-bar { display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin: 0.2rem 0 0.6rem;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06); }
      .wc-bar span { display: block; height: 100%; }
      .wc-legend { display:flex; justify-content:space-between; font-family:'Space Mono',monospace;
                   font-size:0.74rem; color:#8FA399; letter-spacing:0.02em; }

      .wc-verdict {
        font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.14em;
        font-size: 1.05rem; color: #08140F; background: #CCFF00; display: inline-block;
        padding: 0.6rem 1.4rem; border-radius: 999px; font-weight: 700; margin-bottom: 0.4rem;
      }
      .wc-empty {
        border: 1px dashed rgba(255,255,255,0.14); border-radius: 14px; padding: 1.4rem 1.6rem;
        color: #8FA399; font-size: 0.95rem; background: rgba(255,255,255,0.015);
      }
      hr { border-color: rgba(255,255,255,0.07) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plotly(fig, title=None):
    """Apply the broadcast theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#A9BBB1", size=13),
        title=dict(text=title, font=dict(family="Oswald, sans-serif", color="#E8F0EC", size=16)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Model pipeline — thin cached wrappers around the shared module
# (all feature/model logic lives in world_cup_predictor.py; the app only adds
#  Streamlit caching so each step runs once and is shared across visitors)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading match history...")
def load_raw_data():
    return wc.load_raw_data()


@st.cache_data(show_spinner="Engineering features...")
def engineer_features(df):
    return wc.engineer_features(df)


@st.cache_resource(show_spinner="Training XGBoost model... (about 45 seconds)")
def train_model(model_df):
    return wc.train_model(model_df)

# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="wc-hero">
<p class="wc-kicker">For the Deli 🫡</p>
<h1 class="wc-title">Nic's World Cup<br>Match <em>Predictor</em></h1>
<p class="wc-sub">An XGBoost model trained on every international result since 2000. Pick two sides and a match context to read the predicted outcome probabilities.</p>
</div>
    """,
    unsafe_allow_html=True,
)

raw_df = load_raw_data()
model_df, team_log, final_elo = engineer_features(raw_df)
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
        proba = wc.predict_match(model, team_log, final_elo, home_team, away_team, neutral, tourn_importance)

        if proba is None:
            st.error("No form history found for one of these teams — try a different matchup.")
        else:
            p_home, p_draw, p_away = (proba * 100)
            labels = [f"{home_team} win", "Draw", f"{away_team} win"]
            pick_idx = int(np.argmax(proba))
            venue_line = "Neutral venue" if neutral else f"{home_team} hosting"

            st.markdown(
                f"""
<div class="wc-verdict">🏆 {labels[pick_idx]} &nbsp;·&nbsp; {proba[pick_idx]*100:.1f}%</div>
<div class="wc-board">
<div class="wc-side"><div class="name">{home_team}</div><div class="pct" style="color:{HOME}">{p_home:.1f}<span style="font-size:1.4rem">%</span></div></div>
<div class="wc-vs">VS<span class="draw-pct">{p_draw:.0f}%</span>draw</div>
<div class="wc-side"><div class="name">{away_team}</div><div class="pct" style="color:{AWAY}">{p_away:.1f}<span style="font-size:1.4rem">%</span></div></div>
</div>
<div class="wc-bar"><span style="width:{p_home}%;background:{HOME}"></span><span style="width:{p_draw}%;background:{DRAW}"></span><span style="width:{p_away}%;background:{AWAY}"></span></div>
<div class="wc-legend"><span style="color:{HOME}">● {home_team}</span><span>● Draw — {venue_line}</span><span style="color:{AWAY}">{away_team} ●</span></div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="wc-empty">Pick two teams above and hit <b>Predict result</b> to read the odds.</div>',
            unsafe_allow_html=True,
        )

# ── Model performance tab ──────────────────────────────────────────────
with tab_model:
    st.subheader("Holdout test performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['acc']*100:.1f}%")
    c2.metric("Baseline accuracy", f"{metrics['baseline_acc']*100:.1f}%")
    c3.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    c4.metric("Lift over baseline", f"{(metrics['acc']-metrics['baseline_acc'])*100:+.1f} pp")

    c5, c6, _, _ = st.columns(4)
    c5.metric("Draw recall", f"{metrics['draw_recall']*100:.1f}%", help="Share of actual draws the model correctly calls.")
    c6.metric("Log-loss", f"{metrics['log_loss']:.3f}", help="Probability quality — lower is better.")

    st.caption(
        f"Trained on {metrics['n_train']:,} matches up to {metrics['split_date']}, "
        f"tested on {metrics['n_test']:,} matches after that date."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion matrix (row-normalized)")
        fig_cm = px.imshow(
            metrics['cm'], x=wc.CLASS_NAMES, y=wc.CLASS_NAMES,
            color_continuous_scale=[[0, INK], [0.5, "#1F5C3E"], [1, LIME]],
            text_auto=".2f", zmin=0, zmax=1,
            labels=dict(x="Predicted", y="Actual", color="Proportion"),
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(style_plotly(fig_cm), use_container_width=True)

    with col2:
        st.subheader("Feature importance")
        importances = pd.Series(model.feature_importances_, index=wc.FEATURES).sort_values()
        fig_imp = px.bar(
            x=importances.values, y=importances.index, orientation='h',
            labels={'x': 'Importance (gain)', 'y': ''},
        )
        fig_imp.update_traces(marker_color=LIME)
        fig_imp.update_layout(height=400)
        st.plotly_chart(style_plotly(fig_imp), use_container_width=True)

    st.caption("Label scheme: 0 = home win, 1 = draw, 2 = away win.")