#//////////////////////// Loading libraries ////////////////////////////////////////
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Consistent style
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
AXON_BLUE  = '#4A90D9'
AXON_GREEN = '#27AE60'
AXON_RED   = '#E74C3C'

# Label scheme used throughout: 0 = home win, 1 = draw, 2 = away win
CLASS_NAMES = ['Home win', 'Draw', 'Away win']
print('Libraries loaded ✓')


#//////////////////////// Loading the dataset ////////////////////////////////////////
URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
raw = pd.read_csv(URL, parse_dates=['date'])

print(f'Raw shape: {raw.shape}')
print(f"Date range: {raw['date'].min().date()} → {raw['date'].max().date()}")

# Drop matches that haven't been played yet (future fixtures have NaN scores)
df = raw.dropna(subset=['home_score', 'away_score']).copy()
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

# Focus on the modern era so 'recent form' is meaningful and styles are comparable
df = df[df['date'] >= '2000-01-01'].sort_values('date').reset_index(drop=True)
print(f'\nPlayed matches since 2000: {df.shape[0]:,}')


#///////////////////////// Exploratory Data Analysis ////////////////////////////////////
# Derive the result label from the final score
#   0 = home win, 1 = draw, 2 = away win
conditions = [
    df['home_score'] > df['away_score'],
    df['home_score'] == df['away_score'],
]
df['result'] = np.select(conditions, [0, 1], default=2)
df['total_goals'] = df['home_score'] + df['away_score']

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# --- Result distribution ---
counts = df['result'].value_counts().sort_index()
colors = [AXON_GREEN, '#999999', AXON_RED]
axes[0].bar(CLASS_NAMES, counts.values, color=colors, alpha=0.85)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f'{v/len(df)*100:.1f}%', ha='center', fontweight='bold')
axes[0].set_title('Match Results (home perspective)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of matches')

# --- Total goals per match ---
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


#//////////////////////// Feature Engineering /////////////////////////////////////////
# Build a per-team match log: each match -> two rows (home view + away view
WINDOW = 5  # number of previous matches to average over
def build_team_log(matches):
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

df['match_id'] = df.index
team_log = build_team_log(df)
print(f'Team log rows (2 per match): {len(team_log):,}')

g = team_log.groupby('team')
team_log['form_pts'] = g['pts'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
team_log['form_gd'] = g['gd'].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).mean())
team_log['matches_played'] = g.cumcount()

# Sanity check: a team's VERY FIRST match must have NaN form (nothing to average yet).
# If form is non-null on a team's first game, you leaked the current result.
sample = team_log[team_log['team'] == 'Brazil'].head(6)
print(sample[['date', 'team', 'pts', 'gd', 'form_pts', 'form_gd', 'matches_played']].to_string(index=False))
assert team_log.loc[team_log.groupby('team').head(1).index, 'form_pts'].isna().all(), \
    'Leakage! A team\'s first match should have NaN form.'
print('\n✓ Leakage check passed: first match per team has NaN form.')


# Map each side's form back onto the match rows via match_id
home_feats = (team_log[team_log['side'] == 'home']
              .set_index('match_id')[['form_pts', 'form_gd', 'matches_played']]
              .rename(columns=lambda c: 'home_' + c))
away_feats = (team_log[team_log['side'] == 'away']
              .set_index('match_id')[['form_pts', 'form_gd', 'matches_played']]
              .rename(columns=lambda c: 'away_' + c))

data = df.set_index('match_id').join(home_feats).join(away_feats)

# Difference features (home minus away) + match context
data['form_pts_diff'] = data['home_form_pts'] - data['away_form_pts']
data['form_gd_diff']  = data['home_form_gd']  - data['away_form_gd']
data['exp_diff']      = data['home_matches_played'] - data['away_matches_played']

# Home advantage: 1 only when the home team is genuinely hosting (not a neutral venue)
data['is_home_advantage'] = (~data['neutral']).astype(int)
data['is_neutral']        = data['neutral'].astype(int)

# Tournament importance: competitive matches matter more than friendlies
MAJOR = ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'African Cup of Nations',
         'AFC Asian Cup', 'UEFA Nations League', 'CONCACAF Nations League']
def importance(t):
    if t == 'Friendly':
        return 0
    if any(t == m for m in MAJOR):
        return 2          # major finals tournament
    return 1              # qualifier / regional cup
data['tournament_importance'] = data['tournament'].apply(importance)

FEATURES = [
    'form_pts_diff', 'form_gd_diff', 'exp_diff',
    'home_form_pts', 'home_form_gd', 'away_form_pts', 'away_form_gd',
    'is_home_advantage', 'is_neutral', 'tournament_importance',
]

# Drop rows where either team has no prior history yet (form is NaN)
model_df = data.dropna(subset=['home_form_pts', 'away_form_pts']).copy()
print(f'Modelling rows: {len(model_df):,}  |  Features: {len(FEATURES)}')
model_df[FEATURES + ['result']].head()

model_df = model_df.sort_values('date')
X = model_df[FEATURES]
y = model_df['result']

split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

split_date = model_df['date'].iloc[split_idx].date()
print(f'Train: {len(X_train):,} matches (up to {split_date})')
print(f'Test:  {len(X_test):,} matches (from {split_date} onward)')

# Majority-class baseline = always predict the most common training result
majority_class = y_train.mode()[0]
baseline_acc = (y_test == majority_class).mean()
print(f'\nMajority class in train: {CLASS_NAMES[majority_class]}')
print(f'Baseline accuracy (always predict majority): {baseline_acc:.4f}')


#//////////////////////// XGBoost Model /////////////////////////////////////////
model = XGBClassifier(
    objective='multi:softprob', num_class = 3,
    n_estimators=300, max_depth = 4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print('Model trained ✓')
print(f'Trees: {model.n_estimators}  |  Max depth: {model.max_depth}')
print(f'Predicted on {len(y_pred):,} test matches')


#//////////////////////// Model Evaluation /////////////////////////////////////////
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f'Test accuracy:   {acc:.4f}   (baseline: {baseline_acc:.4f})')
print(f'Macro-F1:        {macro_f1:.4f}')
print(f'Lift over baseline: {(acc - baseline_acc) * 100:+.1f} percentage points\n')
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3, width_ratios=[1, 1.1])

# --- Confusion matrix (row-normalised) ---
ax1 = fig.add_subplot(gs[0])
cm = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax1,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False,
            vmin=0, vmax=1)
ax1.set_title(f'Confusion Matrix (row-normalised)\nAccuracy: {acc:.3f}  |  Macro-F1: {macro_f1:.3f}',
              fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# --- Accuracy vs baseline bar ---
ax2 = fig.add_subplot(gs[1])
bars = ax2.bar(['Majority\nbaseline', 'XGBoost'],
               [baseline_acc, acc],
               color=['#999999', AXON_GREEN], alpha=0.9, width=0.55)
for b, v in zip(bars, [baseline_acc, acc]):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}',
             ha='center', fontweight='bold')
ax2.set_ylim(0, max(acc, baseline_acc) * 1.25)
ax2.set_ylabel('Test accuracy')
ax2.set_title('Model vs Baseline', fontweight='bold')

plt.suptitle('World Cup Goal Predictor — Performance', fontsize=14, fontweight='bold', y=1.02)
plt.show()


#Feature Importance 
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


#//////////////////////// Testing Matchups /////////////////////////////////////////
# ✏️  Change these to any two teams in the dataset!
HOME_TEAM = 'Czech Republic'
AWAY_TEAM = 'South Korea'
AT_NEUTRAL_VENUE = True          # True = World Cup-style neutral ground
TOURNAMENT_IMPORTANCE = 2         # 0 = friendly, 1 = qualifier/regional, 2 = major finals

def latest_form(team):
    """Most recent known form for a team from the team log."""
    rows = team_log[team_log['team'] == team].dropna(subset=['form_pts'])
    if rows.empty:
        raise ValueError(f'No form history for "{team}". Check spelling / try another team.')
    last = rows.sort_values('date').iloc[-1]
    return last['form_pts'], last['form_gd'], last['matches_played']

def predict_match(home, away, neutral=False, importance=1):
    h_pts, h_gd, h_exp = latest_form(home)
    a_pts, a_gd, a_exp = latest_form(away)
    row = pd.DataFrame([{
        'form_pts_diff': h_pts - a_pts,
        'form_gd_diff':  h_gd - a_gd,
        'exp_diff':      h_exp - a_exp,
        'home_form_pts': h_pts, 'home_form_gd': h_gd,
        'away_form_pts': a_pts, 'away_form_gd': a_gd,
        'is_home_advantage': 0 if neutral else 1,
        'is_neutral':        1 if neutral else 0,
        'tournament_importance': importance,
    }])[FEATURES]

    proba = model.predict_proba(row)[0]
    venue = 'neutral venue' if neutral else f'{home} at home'
    print('━' * 56)
    print(f'  {home}  vs  {away}   ({venue})')
    print('━' * 56)
    for name, p in zip(CLASS_NAMES, proba):
        label = name if name != 'Home win' else f'{home} win'
        label = label if name != 'Away win' else f'{away} win'
        bar = '█' * int(p * 30) + '░' * (30 - int(p * 30))
        print(f'  {label:<18} {p*100:5.1f}%  [{bar}]')
    print('━' * 56)
    pick = CLASS_NAMES[int(np.argmax(proba))]
    pick = f'{home} win' if pick == 'Home win' else (f'{away} win' if pick == 'Away win' else 'Draw')
    print(f'  🏆 Most likely: {pick}')

predict_match(HOME_TEAM, AWAY_TEAM, neutral=AT_NEUTRAL_VENUE, importance=TOURNAMENT_IMPORTANCE)

# A couple more to compare
print()
predict_match('United States', 'Paraguay', neutral=False, importance=2)
print()
predict_match('Mexico', 'South Africa', neutral=False, importance=2)