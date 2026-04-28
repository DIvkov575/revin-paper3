#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ETT data goes to both baselines
INFORMER_DATA="$REPO_ROOT/baselines/Informer2020/data/ETT"
SCINET_DATA="$REPO_ROOT/baselines/SCINet/data"

mkdir -p "$INFORMER_DATA" "$SCINET_DATA"

ETT_BASE="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

for f in ETTh1.csv ETTh2.csv ETTm1.csv; do
    echo "Downloading $f..."
    wget -q "$ETT_BASE/$f" -O "$INFORMER_DATA/$f"
    cp "$INFORMER_DATA/$f" "$SCINET_DATA/$f"
done

# ECL: download raw UCI data and preprocess to 321-client hourly CSV
ECL_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
TMP="$REPO_ROOT/.ecl_tmp"
mkdir -p "$TMP"

echo "Downloading ECL (raw UCI)..."
wget -q "$ECL_URL" -O "$TMP/LD2011_2014.txt.zip"
unzip -q "$TMP/LD2011_2014.txt.zip" -d "$TMP"

echo "Preprocessing ECL..."
python3 "$REPO_ROOT/preprocess_ecl.py" "$TMP/LD2011_2014.txt" "$INFORMER_DATA/ECL.csv"
cp "$INFORMER_DATA/ECL.csv" "$SCINET_DATA/ECL.csv"

rm -rf "$TMP"
echo "Done. Data written to:"
echo "  $INFORMER_DATA"
echo "  $SCINET_DATA"
