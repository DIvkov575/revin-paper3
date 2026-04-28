"""
plot_rollout.py — qualitative forecast rollout figures

Stitches non-overlapping prediction windows and plots
groundtruth vs baseline vs baseline+RevIN for a chosen feature.

Usage (auto-discover result pairs):
    python plot_rollout.py --model informer --data ETTh1 --pred_len 168
    python plot_rollout.py --model scinet   --data ETTh1 --pred_len 168

Usage (explicit paths):
    python plot_rollout.py \
        --base_dir  baselines/Informer2020/results/RevIN[False]_informer_ETTh1_... \
        --revin_dir baselines/Informer2020/results/RevIN[True]_informer_ETTh1_...  \
        --label "Informer ETTh1 pl=168" --feature 0
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── data helpers ─────────────────────────────────────────────────────────────

def stitch(arr, pred_len):
    """
    arr: (n_windows, pred_len, n_features)
    Returns (T, n_features) by taking every pred_len-th window (non-overlapping).
    """
    n_windows = arr.shape[0]
    step = pred_len
    indices = range(0, n_windows, step)
    segments = [arr[i] for i in indices]
    return np.concatenate(segments, axis=0)   # (T, n_features)


def load_pair(base_dir, revin_dir):
    pred_b = np.load(os.path.join(base_dir,  'pred.npy'))
    true_b = np.load(os.path.join(base_dir,  'true.npy'))
    pred_r = np.load(os.path.join(revin_dir, 'pred.npy'))
    # groundtruth is the same for both — use base version
    pred_len = pred_b.shape[1]
    return (
        stitch(true_b, pred_len),
        stitch(pred_b, pred_len),
        stitch(pred_r, pred_len),
    )


# ── directory discovery ───────────────────────────────────────────────────────

def find_informer_dirs(results_root, data, pred_len):
    # avoid glob mishandling literal brackets by listing + filtering
    all_dirs = [d for d in os.listdir(results_root)
                if f'_informer_{data}_' in d and f'_pl{pred_len}_' in d]
    base  = [d for d in all_dirs if 'RevIN[False]' in d]
    revin = [d for d in all_dirs if 'RevIN[True]'  in d]
    if not base or not revin:
        raise FileNotFoundError(
            f'No matching Informer result dirs for data={data} pl={pred_len}\n'
            f'Available: {all_dirs}'
        )
    return (os.path.join(results_root, base[0]),
            os.path.join(results_root, revin[0]))


def find_scinet_dirs(results_root, data, pred_len):
    pattern = os.path.join(results_root, f'SCINet_{data}_ftM_*_pl{pred_len}_*')
    dirs = glob.glob(pattern)
    base  = [d for d in dirs if 'oursTrue'  not in d]
    revin = [d for d in dirs if 'oursTrue'  in d]
    if not base or not revin:
        raise FileNotFoundError(
            f'Could not find matching SCINet result dirs for data={data} pl={pred_len}\n'
            f'Searched: {pattern}\nFound: {dirs}'
        )
    return base[0], revin[0]


# ── plotting ─────────────────────────────────────────────────────────────────

FEATURE_NAMES = {
    'ETTh1': ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
    'ETTh2': ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
    'ETTm1': ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
    'ECL':   [f'MT_{i+1:03d}' for i in range(321)],
}


def plot_rollout(true, pred_base, pred_revin, label, feature, model_name, out_path,
                 context_csv=None, context_col=None, train_end_idx=None):
    feat_vals = true[:, feature]
    base_vals = pred_base[:, feature]
    rev_vals  = pred_revin[:, feature]

    fig, ax = plt.subplots(figsize=(14, 4))

    t_test = np.arange(len(feat_vals))

    if context_csv is not None and os.path.exists(context_csv):
        import pandas as pd
        df = pd.read_csv(context_csv)
        col = context_col or df.columns[feature + 1]   # +1 for date column
        ctx_len = min(500, train_end_idx or len(df))
        ctx_start = (train_end_idx or len(df)) - ctx_len
        ctx = df[col].values[ctx_start:train_end_idx]
        t_ctx = np.arange(-len(ctx), 0)
        ax.plot(t_ctx, ctx, color='#E84A5F', lw=1.2, alpha=0.7, label='Context (train)')
        ax.axvline(0, color='#555', lw=1, ls='--')

    ax.plot(t_test, feat_vals, color='#E84A5F', lw=1.5, label='Groundtruth')
    ax.plot(t_test, base_vals, color='#F7A541', lw=1.3, alpha=0.85, label=model_name)
    ax.plot(t_test, rev_vals,  color='#44B0FF', lw=1.3, alpha=0.85, label=f'{model_name} + RevIN')

    ax.set_title(label, fontsize=11)
    ax.set_xlabel('Timestep')
    ax.legend(fontsize=9, loc='upper left')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved → {out_path}')


def multi_feature_plot(true, pred_base, pred_revin, features, feature_names,
                       label, model_name, out_path):
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(true.shape[0])
    for ax, feat in zip(axes, features):
        name = feature_names[feat] if feat < len(feature_names) else f'feature {feat}'
        ax.plot(t, true[:, feat],       color='#E84A5F', lw=1.5, label='Groundtruth')
        ax.plot(t, pred_base[:, feat],  color='#F7A541', lw=1.3, alpha=0.85, label=model_name)
        ax.plot(t, pred_revin[:, feat], color='#44B0FF', lw=1.3, alpha=0.85, label=f'{model_name} + RevIN')
        ax.set_ylabel(name, fontsize=9)
        ax.spines[['top','right']].set_visible(False)

    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].set_title(label, fontsize=11)
    axes[-1].set_xlabel('Timestep (test set)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved → {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # auto-discovery mode
    parser.add_argument('--model',    default='informer', choices=['informer', 'scinet'])
    parser.add_argument('--data',     default='ETTh1')
    parser.add_argument('--pred_len', type=int, default=168)
    # explicit path mode
    parser.add_argument('--base_dir',  default=None)
    parser.add_argument('--revin_dir', default=None)
    parser.add_argument('--label',     default=None)
    # plot options
    parser.add_argument('--feature',   type=int, default=None,
                        help='single feature index; omit to plot all')
    parser.add_argument('--out_dir',   default='figures')
    args = parser.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(repo, args.out_dir), exist_ok=True)

    # locate result directories
    if args.base_dir and args.revin_dir:
        base_dir  = args.base_dir
        revin_dir = args.revin_dir
        model_name = args.model.capitalize()
        label = args.label or f'{model_name} {args.data} pl={args.pred_len}'
    else:
        if args.model == 'informer':
            results_root = os.path.join(repo, 'baselines/Informer2020/results')
            base_dir, revin_dir = find_informer_dirs(results_root, args.data, args.pred_len)
            model_name = 'Informer'
        else:
            results_root = os.path.join(repo, 'baselines/SCINet/exp/ett_results')
            base_dir, revin_dir = find_scinet_dirs(results_root, args.data, args.pred_len)
            model_name = 'SCINet'
        label = f'{model_name} — {args.data}  pred_len={args.pred_len}'

    print(f'Base:  {base_dir}')
    print(f'RevIN: {revin_dir}')

    true, pred_base, pred_revin = load_pair(base_dir, revin_dir)
    print(f'Stitched shape: true={true.shape} base={pred_base.shape} revin={pred_revin.shape}')

    feat_names = FEATURE_NAMES.get(args.data, [f'f{i}' for i in range(true.shape[1])])
    tag = f'{args.model}_{args.data}_pl{args.pred_len}'

    if args.feature is not None:
        out = os.path.join(repo, args.out_dir, f'{tag}_feat{args.feature}.png')
        plot_rollout(true, pred_base, pred_revin,
                     label=f'{label}  [{feat_names[args.feature]}]',
                     feature=args.feature,
                     model_name=model_name,
                     out_path=out)
    else:
        # plot all features in one multi-panel figure
        features = list(range(true.shape[-1]))
        out = os.path.join(repo, args.out_dir, f'{tag}_all_features.png')
        multi_feature_plot(true, pred_base, pred_revin,
                           features=features,
                           feature_names=feat_names,
                           label=label,
                           model_name=model_name,
                           out_path=out)


if __name__ == '__main__':
    main()
