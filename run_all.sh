#!/usr/bin/env bash
# Reproduces Table 1 (ETTh1) and Table 2 (ETTh1 fixed input=48)
# for Informer and SCINet, with and without RevIN.
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/run.log"; }

# ── Informer ────────────────────────────────────────────────────────────────
run_informer() {
    local tag="$1"; shift
    log "Informer  $tag"
    (cd "$REPO/baselines/Informer2020" && python3 -u main_informer.py "$@" --itr 1 2>&1) \
        | tee -a "$LOG_DIR/informer_${tag}.log"
    rm -rf "$REPO/baselines/Informer2020/checkpoints/" || true
}

COMMON_INF="--model informer --features M --e_layers 2 --d_layers 1 --attn prob --des Exp"

# Table 1 — ETTh1
for pl in 24 48 168 336 720 960; do
    case $pl in
        24)  sl=48;  ll=48  ;;
        48)  sl=96;  ll=48  ;;
        168) sl=168; ll=168 ;;
        336) sl=168; ll=168 ;;
        720) sl=336; ll=336 ;;
        960) sl=336; ll=336 ;;
    esac
    run_informer "ETTh1_pl${pl}"       $COMMON_INF --data ETTh1 --seq_len $sl --label_len $ll --pred_len $pl
    run_informer "ETTh1_RevIN_pl${pl}" $COMMON_INF --data ETTh1 --seq_len $sl --label_len $ll --pred_len $pl --use_RevIN
done

# Table 2 — ETTh1 fixed seq_len=48
log "=== Table 2: Informer ETTh1 fixed input=48 ==="
for pl in 48 168 336 720 960; do
    run_informer "T2_ETTh1_pl${pl}"       $COMMON_INF --data ETTh1 --seq_len 48 --label_len 48 --pred_len $pl
    run_informer "T2_ETTh1_RevIN_pl${pl}" $COMMON_INF --data ETTh1 --seq_len 48 --label_len 48 --pred_len $pl --use_RevIN
done

log "=== Informer done ==="

# ── SCINet ───────────────────────────────────────────────────────────────────
run_scinet() {
    local tag="$1"; shift
    log "SCINet  $tag"
    (cd "$REPO/baselines/SCINet" && python3 -u run_ETTh.py --save "$@" --seed 42 --itr 1 2>&1) \
        | tee -a "$LOG_DIR/scinet_${tag}.log"
    rm -rf "$REPO/baselines/SCINet/exp/ETT_checkpoints/" || true
}

# Table 1 — ETTh1
run_scinet "ETTh1_pl24"        --data ETTh1 --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 4 --stacks 1 --levels 3 --lr 3e-3 --batch_size 8   --dropout 0.5
run_scinet "ETTh1_RevIN_pl24"  --data ETTh1 --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 4 --stacks 1 --levels 3 --lr 3e-3 --batch_size 8   --dropout 0.5  --ours
run_scinet "ETTh1_pl48"        --data ETTh1 --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 4 --stacks 1 --levels 3 --lr 9e-3 --batch_size 16  --dropout 0.25
run_scinet "ETTh1_RevIN_pl48"  --data ETTh1 --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 4 --stacks 1 --levels 3 --lr 9e-3 --batch_size 16  --dropout 0.25 --ours
run_scinet "ETTh1_pl168"       --data ETTh1 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --levels 3 --lr 5e-4 --batch_size 32  --dropout 0.5
run_scinet "ETTh1_RevIN_pl168" --data ETTh1 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --levels 3 --lr 5e-4 --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTh1_pl336"       --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5
run_scinet "ETTh1_RevIN_pl336" --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5  --ours
run_scinet "ETTh1_pl720"       --data ETTh1 --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 5 --lr 5e-5 --batch_size 256 --dropout 0.5
run_scinet "ETTh1_RevIN_pl720" --data ETTh1 --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 5 --lr 5e-5 --batch_size 256 --dropout 0.5  --ours
run_scinet "ETTh1_pl960"       --data ETTh1 --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5
run_scinet "ETTh1_RevIN_pl960" --data ETTh1 --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5  --ours

log "=== All done ==="
