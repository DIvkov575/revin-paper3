"""
Preprocess the UCI ElectricityLoadDiagrams20112014 dataset into the
321-client hourly ECL.csv used by Informer / SCINet.

Raw file: semicolon-separated, 15-min resolution, 370 clients (MT_001..MT_370).
Output:   comma-separated, hourly, 321 clients (MT_001..MT_321), date column first.
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]

print(f"Reading {src} ...")
df = pd.read_csv(src, sep=';', index_col=0, decimal=',')
df.index = pd.to_datetime(df.index)

# resample 15-min → hourly by summing (kWh per hour)
df = df.resample('1h').sum()

# keep first 321 clients to match the paper
df = df.iloc[:, :321]
df.columns = [f'MT_{i+1:03d}' for i in range(321)]

df.index.name = 'date'
df = df.reset_index()

df.to_csv(dst, index=False)
print(f"Wrote {dst}  shape={df.shape}")
