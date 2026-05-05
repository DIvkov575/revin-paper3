"""Plot Nasdaq rollout: training context + one 168-day prediction window."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, 'baselines/Informer2020/results')
DATA    = os.path.join(REPO, 'baselines/Informer2020/data/ETT/nasdaq.csv')

BASE_DIR  = os.path.join(RESULTS, 'RevIN[False]_informer_custom_ftM_sl60_ll60_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0')
REVIN_DIR = os.path.join(RESULTS, 'RevIN[True]_informer_custom_ftM_sl60_ll60_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0')

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, parse_dates=['date'])
feat_cols = [c for c in df.columns if c != 'date']

# same split as Informer's Dataset_Custom: 70/10/20
n = len(df)
train_end = int(n * 0.7)   # last training index

# column indices for the 3 features we want to plot
close_idx = feat_cols.index('Close')
dtb6_idx  = feat_cols.index('DTB6')
de1_idx   = feat_cols.index('DE1')

# first 168-step prediction window (window 0 of the test set)
pred_base  = np.load(os.path.join(BASE_DIR,  'pred.npy'))[0]   # (168, 82)
pred_revin = np.load(os.path.join(REVIN_DIR, 'pred.npy'))[0]
true_vals  = np.load(os.path.join(BASE_DIR,  'true.npy'))[0]

# ── plot ──────────────────────────────────────────────────────────────────────
features = [
    ('Close', close_idx),
    ('DTB6',  dtb6_idx),
    ('DE1',   de1_idx),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

for ax, (name, idx) in zip(axes, features):
    # training context — show last 300 days of training data
    ctx_start = max(0, train_end - 300)
    ctx_dates = df['date'].values[ctx_start:train_end]
    ctx_vals  = df[name].values[ctx_start:train_end]

    # test dates for the 168-step window
    test_dates = df['date'].values[train_end:train_end + 168]

    ax.plot(ctx_dates, ctx_vals,
            color='#E84A5F', lw=1.2, alpha=0.8, label='Context')
    ax.axvline(df['date'].values[train_end], color='#555', lw=1, ls='--')
    ax.plot(test_dates, true_vals[:len(test_dates), idx],
            color='#E84A5F', lw=1.5, label='Groundtruth')
    ax.plot(test_dates, pred_base[:len(test_dates), idx],
            color='#F7A541', lw=1.3, alpha=0.9, label='Informer')
    ax.plot(test_dates, pred_revin[:len(test_dates), idx],
            color='#44B0FF', lw=1.3, alpha=0.9, label='Informer + RevIN')

    ax.set_ylabel(name, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].legend(fontsize=9, loc='upper left')
axes[0].set_title('Nasdaq — Informer vs Informer+RevIN  (pred_len=168)', fontsize=11)
axes[-1].set_xlabel('Date')

plt.tight_layout()
os.makedirs(os.path.join(REPO, 'figures'), exist_ok=True)
out = os.path.join(REPO, 'figures/nasdaq_rollout.png')
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved → {out}')
