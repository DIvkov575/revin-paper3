"""Compute MAE/MSE directly from pred.npy and true.npy — Table 1 only."""
import os
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'baselines/Informer2020/results')

# Table 1 seq_len mapping for ETTh1
TABLE1_SL = {24: 48, 48: 96, 168: 168, 336: 168, 720: 336, 960: 336}

def compute(path):
    pred = np.load(os.path.join(path, 'pred.npy'))
    true = np.load(os.path.join(path, 'true.npy'))
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    return mae, mse

print(f"{'pl':>6} {'MAE':>8} {'MAE+RevIN':>11} {'ΔMAE':>8}  {'MSE':>8} {'MSE+RevIN':>11} {'ΔMSE':>8}")
print("-" * 70)

dirs = os.listdir(RESULTS) if os.path.exists(RESULTS) else []

for pl, sl in TABLE1_SL.items():
    base_dir  = next((d for d in dirs
                      if 'RevIN[False]' in d and f'_pl{pl}_' in d and f'_sl{sl}_' in d), None)
    revin_dir = next((d for d in dirs
                      if 'RevIN[True]'  in d and f'_pl{pl}_' in d and f'_sl{sl}_' in d), None)

    b_mae = b_mse = r_mae = r_mse = None
    if base_dir:
        b_mae, b_mse = compute(os.path.join(RESULTS, base_dir))
    if revin_dir:
        r_mae, r_mse = compute(os.path.join(RESULTS, revin_dir))

    b_mae_s = f'{b_mae:.4f}' if b_mae is not None else '—'
    r_mae_s = f'{r_mae:.4f}' if r_mae is not None else '—'
    d_mae_s = f'{r_mae-b_mae:+.4f}' if (b_mae and r_mae) else '—'
    b_mse_s = f'{b_mse:.4f}' if b_mse is not None else '—'
    r_mse_s = f'{r_mse:.4f}' if r_mse is not None else '—'
    d_mse_s = f'{r_mse-b_mse:+.4f}' if (b_mse and r_mse) else '—'

    print(f"{pl:>6} {b_mae_s:>8} {r_mae_s:>11} {d_mae_s:>8}  {b_mse_s:>8} {r_mse_s:>11} {d_mse_s:>8}")
