#//////////////////////// Loading libraries ////////////////////////////////////////
# Plain (no streamlit) so this file can be imported by streamlitApp.py AND run
# directly as a script. Charting/EDA libraries (matplotlib, seaborn) are imported
# lazily inside the __main__ block so importing this module stays light and never
# pulls in dependencies the deployed app doesn't have.
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

# Label scheme used throughout: 0 = home win, 1 = draw, 2 = away win
CLASS_NAMES = ['Home win', 'Draw', 'Away win']
WINDOW = 5            # number of previous matches to average over for form
DRAW_BOOST = 1.4      # sample-weight multiplier for the draw class (1.0 = off)
DATA_URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'

MAJOR = ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'African Cup of Nations',
         'AFC Asian Cup', 'UEFA Nations League', 'CONCACAF Nations League']

FEATURES = [
    'form_pts_diff', 'form_gd_diff', 'exp_diff',
    'home_form_pts', 'home_form_gd', 'away_form_pts', 'away_form_gd',
    'home_elo', 'away_elo', 'elo_diff',
    'home_form_gf', 'home_form_ga', 'away_form_gf', 'away_form_ga',
    'form_gf_diff', 'form_ga_diff',
    'is_home_advantage', 'is_neutral', 'tournament_importance',
]


#//////////////////////// Data loading ////////////////////////////////////////
def load_raw_data():
    """Download played international matches since 2000, sorted by date."""
    raw = pd.read_csv(DATA_URL, parse_dates=['date'])
    # Drop matches that haven't been played yet (future fixtures have NaN scores)
    df = raw.dropna(subset=['home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    # Focus on the modern era so 'recent form' is meaningful
    df = df[df['date'] >= '2000-01-01'].sort_values('date').reset_index(drop=True)
    return df


#//////////////////////// Feature Engineering /////////////////////////////////////////
def build_team_log(matches):
    """Per-team match log: each match -> two rows (home view + away view)."""
    sides = {
        'home': ('home_team', 'home_score', 'away_score'),
        'away': ('away_team', 'away_score', 'home_score'),}
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


def add_elo(matches, k=30, home_adv=65, base=1500):
    """Append pre-match home_elo/away_elo/elo_diff. Returns (matches, final_elo dict).
    `matches` MUST be chronologically sorted."""
    elo = {}
    h_pre, a_pre = [], []
    for r in matches.itertuples():
        e_home = elo.get(r.home_team, base)
        e_away = elo.get(r.away_team, base)
        h_pre.append(e_home)
        a_pre.append(e_away)
        adv = 0 if r.neutral else home_adv
        exp_h = 1 / (1 + 10 ** ((e_away - (e_home + adv)) / 400))
        sh = 1.0 if r.home_score > r.away_score else 0.5 if r.home_score == r.away_score else 0.0
        margin = 1 + np.log1p(abs(r.home_score - r.away_score))  # goal-margin weighting
        elo[r.home_team] = e_home + k * margin * (sh - exp_h)
        elo[r.away_team] = e_away + k * margin * ((1 - sh) - (1 - exp_h))
    matches = matches.copy()
    matches['home_elo'] = h_pre
    matches['away_elo'] = a_pre
    matches['elo_diff'] = matches['home_elo'] - matches['away_elo']
    return matches, elo


def importance(t):
    """Tournament importance: competitive matches matter more than friendlies."""
    if t == 'Friendly':
        return 0
    if any(t == m for m in MAJOR):
        return 2          # major finals tournament
    return 1              # qualifier / regional cup


def engineer_features(df):
    """Build the model table. Returns (model_df, team_log, final_elo)."""
    df = df.copy()
    conditions = [df['home_score'] > df['away_score'], df['home_score'] == df['away_score']]
    df['result'] = np.select(conditions, [0, 1], default=2)
    df['match_id'] = df.index

    df, final_elo = add_elo(df)

    team_log = build_team_log(df)
    g = team_log.groupby('team')
    team_log['form_pts'] = g['pts'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
    team_log['form_gd'] = g['gd'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
    team_log['form_gf'] = g['goals_for'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
    team_log['form_ga'] = g['goals_against'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
    team_log['matches_played'] = g.cumcount()

    # Map each side's form back onto the match rows via match_id
    form_cols = ['form_pts', 'form_gd', 'form_gf', 'form_ga', 'matches_played']
    home_feats = (team_log[team_log['side'] == 'home']
                  .set_index('match_id')[form_cols]
                  .rename(columns=lambda c: 'home_' + c))
    away_feats = (team_log[team_log['side'] == 'away']
                  .set_index('match_id')[form_cols]
                  .rename(columns=lambda c: 'away_' + c))

    data = df.set_index('match_id').join(home_feats).join(away_feats)

    # Difference features (home minus away) + match context
    data['form_pts_diff'] = data['home_form_pts'] - data['away_form_pts']
    data['form_gd_diff']  = data['home_form_gd']  - data['away_form_gd']
    data['form_gf_diff']  = data['home_form_gf']  - data['away_form_gf']
    data['form_ga_diff']  = data['home_form_ga']  - data['away_form_ga']
    data['exp_diff']      = data['home_matches_played'] - data['away_matches_played']
    data['is_home_advantage'] = (~data['neutral']).astype(int)
    data['is_neutral']        = data['neutral'].astype(int)
    data['tournament_importance'] = data['tournament'].apply(importance)

    # Drop rows where either team has no prior history yet (form is NaN)
    model_df = data.dropna(subset=['home_form_pts', 'away_form_pts']).copy()
    model_df = model_df.sort_values('date')
    return model_df, team_log, final_elo


#//////////////////////// Model training /////////////////////////////////////////
def train_model(model_df):
    """Time-split train + evaluate. Returns (model, metrics).
    `metrics` carries the display scalars used by the Streamlit app plus the raw
    y_test/y_pred/y_proba arrays used by this file's __main__ report/plots."""
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
        subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='mlogloss')

    weights = compute_sample_weight({0: 1.0, 1: DRAW_BOOST, 2: 1.0}, y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    draw_recall = cm[1, 1]  # row-normalized → diagonal is per-class recall
    logloss = log_loss(y_test, y_proba, labels=[0, 1, 2])

    metrics = {
        'baseline_acc': baseline_acc,
        'acc': acc,
        'macro_f1': macro_f1,
        'draw_recall': draw_recall,
        'log_loss': logloss,
        'cm': cm,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'split_date': model_df['date'].iloc[split_idx].date(),
        'majority_class': majority_class,
        # raw arrays for the __main__ classification report / plots (app ignores these)
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }
    return model, metrics


#//////////////////////// Prediction /////////////////////////////////////////
def latest_form(team_log, team):
    """Most recent known form for a team from the team log, or None if no history."""
    rows = team_log[team_log['team'] == team].dropna(subset=['form_pts'])
    if rows.empty:
        return None
    last = rows.sort_values('date').iloc[-1]
    return (last['form_pts'], last['form_gd'], last['form_gf'],
            last['form_ga'], last['matches_played'])


def predict_match(model, team_log, final_elo, home, away, neutral, tourn_importance):
    """Return [P(home win), P(draw), P(away win)] or None if a team has no history."""
    h = latest_form(team_log, home)
    a = latest_form(team_log, away)
    if h is None or a is None:
        return None

    h_pts, h_gd, h_gf, h_ga, h_exp = h
    a_pts, a_gd, a_gf, a_ga, a_exp = a
    h_elo = final_elo.get(home, 1500)
    a_elo = final_elo.get(away, 1500)

    row = pd.DataFrame([{
        'form_pts_diff': h_pts - a_pts,
        'form_gd_diff':  h_gd - a_gd,
        'exp_diff':      h_exp - a_exp,
        'home_form_pts': h_pts, 'home_form_gd': h_gd,
        'away_form_pts': a_pts, 'away_form_gd': a_gd,
        'home_elo': h_elo, 'away_elo': a_elo, 'elo_diff': h_elo - a_elo,
        'home_form_gf': h_gf, 'home_form_ga': h_ga,
        'away_form_gf': a_gf, 'away_form_ga': a_ga,
        'form_gf_diff':  h_gf - a_gf, 'form_ga_diff': h_ga - a_ga,
        'is_home_advantage': 0 if neutral else 1,
        'is_neutral':        1 if neutral else 0,
        'tournament_importance': tourn_importance,
    }])[FEATURES]

    return model.predict_proba(row)[0]


#//////////////////////// Local analysis script ////////////////////////////////////////
# Everything below runs ONLY when executing this file directly
# (`python world_cup_predictor.py`). It is never touched on `import`, so the
# Streamlit app never pulls in matplotlib/seaborn or triggers a download/training.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from sklearn.metrics import classification_report

    plt.rcParams.update({
        'figure.dpi': 120,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    AXON_BLUE  = '#4A90D9'
    AXON_GREEN = '#27AE60'
    AXON_RED   = '#E74C3C'
    print('Libraries loaded ✓')

    # ---- Load ----
    df = load_raw_data()
    print(f'\nPlayed matches since 2000: {df.shape[0]:,}')
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ---- Exploratory Data Analysis ----
    conditions = [df['home_score'] > df['away_score'], df['home_score'] == df['away_score']]
    df['result'] = np.select(conditions, [0, 1], default=2)
    df['total_goals'] = df['home_score'] + df['away_score']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    counts = df['result'].value_counts().sort_index()
    colors = [AXON_GREEN, '#999999', AXON_RED]
    axes[0].bar(CLASS_NAMES, counts.values, color=colors, alpha=0.85)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f'{v/len(df)*100:.1f}%', ha='center', fontweight='bold')
    axes[0].set_title('Match Results (home perspective)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of matches')

    axes[1].hist(df['total_goals'], bins=range(0, 13), color=AXON_BLUE, alpha=0.8,
                 align='left', rwidth=0.85)
    axes[1].set_title('Total Goals per Match', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Goals (home + away)')
    axes[1].set_ylabel('Number of matches')
    axes[1].set_xticks(range(0, 13))
    plt.suptitle('International Football: Results & Goals (since 2000)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    print('Result balance (this is our baseline to beat):')
    print((df['result'].value_counts(normalize=True).sort_index()
           .rename(index=dict(enumerate(CLASS_NAMES))) * 100).round(1).astype(str) + ' %')

    # ---- Feature engineering ----
    model_df, team_log, final_elo = engineer_features(df)
    print(f'\nTeam log rows (2 per match): {len(team_log):,}')
    print(f'Modelling rows: {len(model_df):,}  |  Features: {len(FEATURES)}')

    # Leakage check: a team's VERY FIRST match must have NaN form.
    assert team_log.loc[team_log.groupby('team').head(1).index, 'form_pts'].isna().all(), \
        'Leakage! A team\'s first match should have NaN form.'
    print('✓ Leakage check passed: first match per team has NaN form.')

    # ---- Train + evaluate ----
    model, m = train_model(model_df)
    print(f"\nTrain: {m['n_train']:,} matches (up to {m['split_date']})")
    print(f"Test:  {m['n_test']:,} matches (from {m['split_date']} onward)")
    print(f"Majority class in train: {CLASS_NAMES[m['majority_class']]}")
    print(f"Baseline accuracy (always predict majority): {m['baseline_acc']:.4f}\n")
    print('Model trained ✓')
    print(f"Test accuracy:   {m['acc']:.4f}   (baseline: {m['baseline_acc']:.4f})")
    print(f"Macro-F1:        {m['macro_f1']:.4f}")
    print(f"Log-loss:        {m['log_loss']:.4f}   (lower = better-calibrated probabilities)")
    print(f"Lift over baseline: {(m['acc'] - m['baseline_acc']) * 100:+.1f} percentage points\n")
    print(classification_report(m['y_test'], m['y_pred'], target_names=CLASS_NAMES, digits=3))

    # Confusion matrix + accuracy-vs-baseline bar
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3, width_ratios=[1, 1.1])
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(m['cm'], annot=True, fmt='.2f', cmap='Blues', ax=ax1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False,
                vmin=0, vmax=1)
    ax1.set_title(f"Confusion Matrix (row-normalised)\nAccuracy: {m['acc']:.3f}  |  Macro-F1: {m['macro_f1']:.3f}",
                  fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(['Majority\nbaseline', 'XGBoost'], [m['baseline_acc'], m['acc']],
                   color=['#999999', AXON_GREEN], alpha=0.9, width=0.55)
    for b, v in zip(bars, [m['baseline_acc'], m['acc']]):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}',
                 ha='center', fontweight='bold')
    ax2.set_ylim(0, max(m['acc'], m['baseline_acc']) * 1.25)
    ax2.set_ylabel('Test accuracy')
    ax2.set_title('Model vs Baseline', fontweight='bold')
    plt.suptitle('World Cup Goal Predictor — Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.show()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importances.index, importances.values, color=AXON_BLUE, alpha=0.85)
    ax.set_title('XGBoost Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance (gain)')
    plt.tight_layout()
    plt.show()

    print('Most predictive feature:', importances.idxmax())
    print('\nFull ranking:')
    print(importances.sort_values(ascending=False).round(3).to_string())

    # ---- Testing matchups ----
    def print_prediction(home, away, neutral=False, tourn_importance=1):
        proba = predict_match(model, team_log, final_elo, home, away, neutral, tourn_importance)
        venue = 'neutral venue' if neutral else f'{home} at home'
        print('━' * 56)
        print(f'  {home}  vs  {away}   ({venue})')
        print('━' * 56)
        if proba is None:
            print('  No form history for one of these teams.')
            return
        for name, p in zip(CLASS_NAMES, proba):
            label = f'{home} win' if name == 'Home win' else (f'{away} win' if name == 'Away win' else 'Draw')
            bar = '█' * int(p * 30) + '░' * (30 - int(p * 30))
            print(f'  {label:<18} {p*100:5.1f}%  [{bar}]')
        print('━' * 56)
        pick = CLASS_NAMES[int(np.argmax(proba))]
        pick = f'{home} win' if pick == 'Home win' else (f'{away} win' if pick == 'Away win' else 'Draw')
        print(f'  🏆 Most likely: {pick}')

    print()
    print_prediction('Czech Republic', 'South Korea', neutral=True, tourn_importance=2)
    print()
    print_prediction('United States', 'Paraguay', neutral=False, tourn_importance=2)
    print()
    print_prediction('Mexico', 'South Africa', neutral=False, tourn_importance=2)
