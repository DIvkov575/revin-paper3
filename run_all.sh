#!/usr/bin/env bash
# Reproduces Table 1 (ETTh1/ETTh2/ETTm1/ECL) and Table 2 (ETTh1 fixed input=48)
# for Informer and SCINet, with and without RevIN.
# --itr 1 / single seed for speed; bump to 5 seeds for paper-quality averages.
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
    run_informer "ETTh1_pl${pl}"         $COMMON_INF --data ETTh1 --seq_len $sl --label_len $ll --pred_len $pl
    run_informer "ETTh1_RevIN_pl${pl}"   $COMMON_INF --data ETTh1 --seq_len $sl --label_len $ll --pred_len $pl --use_RevIN
done

# Table 1 — ETTh2
for pl in 24 48 168 336 720 960; do
    case $pl in
        24)  sl=48;  ll=48  ;;
        48)  sl=96;  ll=48  ;;
        168) sl=168; ll=168 ;;
        336) sl=168; ll=168 ;;
        720) sl=336; ll=336 ;;
        960) sl=336; ll=336 ;;
    esac
    run_informer "ETTh2_pl${pl}"         $COMMON_INF --data ETTh2 --seq_len $sl --label_len $ll --pred_len $pl
    run_informer "ETTh2_RevIN_pl${pl}"   $COMMON_INF --data ETTh2 --seq_len $sl --label_len $ll --pred_len $pl --use_RevIN
done

# Table 1 — ETTm1
for pl in 24 48 96 288 672 1344; do
    case $pl in
        24)   sl=48;  ll=24  ;;
        48)   sl=96;  ll=48  ;;
        96)   sl=192; ll=96  ;;
        288)  sl=288; ll=288 ;;
        672)  sl=672; ll=336 ;;
        1344) sl=672; ll=672 ;;
    esac
    run_informer "ETTm1_pl${pl}"         $COMMON_INF --data ETTm1 --freq t --seq_len $sl --label_len $ll --pred_len $pl
    run_informer "ETTm1_RevIN_pl${pl}"   $COMMON_INF --data ETTm1 --freq t --seq_len $sl --label_len $ll --pred_len $pl --use_RevIN
done

# Table 1 — ECL
for pl in 24 48 168 336 720 960; do
    case $pl in
        24)  sl=48;  ll=48  ;;
        48)  sl=96;  ll=48  ;;
        168) sl=168; ll=168 ;;
        336) sl=336; ll=168 ;;
        720) sl=720; ll=336 ;;
        960) sl=720; ll=336 ;;
    esac
    run_informer "ECL_pl${pl}"           $COMMON_INF --data ECL --seq_len $sl --label_len $ll --pred_len $pl
    run_informer "ECL_RevIN_pl${pl}"     $COMMON_INF --data ECL --seq_len $sl --label_len $ll --pred_len $pl --use_RevIN
done

# Table 2 — ETTh1 fixed seq_len=48
log "=== Table 2: Informer ETTh1 fixed input=48 ==="
for pl in 48 168 336 720 960; do
    run_informer "T2_ETTh1_pl${pl}"        $COMMON_INF --data ETTh1 --seq_len 48 --label_len 48 --pred_len $pl
    run_informer "T2_ETTh1_RevIN_pl${pl}"  $COMMON_INF --data ETTh1 --seq_len 48 --label_len 48 --pred_len $pl --use_RevIN
done

log "=== Informer done ==="

# ── SCINet ───────────────────────────────────────────────────────────────────
run_scinet() {
    local tag="$1"; shift
    log "SCINet  $tag"
    (cd "$REPO/baselines/SCINet" && python3 -u run_ETTh.py --save "$@" --seed 42 --itr 1 2>&1) \
        | tee -a "$LOG_DIR/scinet_${tag}.log"
}

# Table 1 — ETTh1
run_scinet "ETTh1_pl24"          --data ETTh1 --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 4 --stacks 1 --levels 3 --lr 3e-3   --batch_size 8   --dropout 0.5
run_scinet "ETTh1_RevIN_pl24"    --data ETTh1 --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 4 --stacks 1 --levels 3 --lr 3e-3   --batch_size 8   --dropout 0.5  --ours
run_scinet "ETTh1_pl48"          --data ETTh1 --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 4 --stacks 1 --levels 3 --lr 9e-3   --batch_size 16  --dropout 0.25
run_scinet "ETTh1_RevIN_pl48"    --data ETTh1 --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 4 --stacks 1 --levels 3 --lr 9e-3   --batch_size 16  --dropout 0.25 --ours
run_scinet "ETTh1_pl168"         --data ETTh1 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --levels 3 --lr 5e-4   --batch_size 32  --dropout 0.5
run_scinet "ETTh1_RevIN_pl168"   --data ETTh1 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --levels 3 --lr 5e-4   --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTh1_pl336"         --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4   --batch_size 512 --dropout 0.5
run_scinet "ETTh1_RevIN_pl336"   --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4   --batch_size 512 --dropout 0.5  --ours
run_scinet "ETTh1_pl720"         --data ETTh1 --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 5 --lr 5e-5   --batch_size 256 --dropout 0.5
run_scinet "ETTh1_RevIN_pl720"   --data ETTh1 --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 5 --lr 5e-5   --batch_size 256 --dropout 0.5  --ours
run_scinet "ETTh1_pl960"         --data ETTh1 --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4   --batch_size 512 --dropout 0.5
run_scinet "ETTh1_RevIN_pl960"   --data ETTh1 --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4   --batch_size 512 --dropout 0.5  --ours

# Table 1 — ETTh2
run_scinet "ETTh2_pl24"          --data ETTh2 --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 8   --stacks 1 --levels 3 --lr 7e-3  --batch_size 16  --dropout 0.25
run_scinet "ETTh2_RevIN_pl24"    --data ETTh2 --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 8   --stacks 1 --levels 3 --lr 7e-3  --batch_size 16  --dropout 0.25 --ours
run_scinet "ETTh2_pl48"          --data ETTh2 --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 4   --stacks 1 --levels 4 --lr 7e-3  --batch_size 4   --dropout 0.5
run_scinet "ETTh2_RevIN_pl48"    --data ETTh2 --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 4   --stacks 1 --levels 4 --lr 7e-3  --batch_size 4   --dropout 0.5  --ours
run_scinet "ETTh2_pl168"         --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 0.5 --stacks 1 --levels 4 --lr 5e-5  --batch_size 16  --dropout 0.5
run_scinet "ETTh2_RevIN_pl168"   --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 0.5 --stacks 1 --levels 4 --lr 5e-5  --batch_size 16  --dropout 0.5  --ours
run_scinet "ETTh2_pl336"         --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1   --stacks 1 --levels 4 --lr 5e-5  --batch_size 128 --dropout 0.5
run_scinet "ETTh2_RevIN_pl336"   --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1   --stacks 1 --levels 4 --lr 5e-5  --batch_size 128 --dropout 0.5  --ours
run_scinet "ETTh2_pl720"         --data ETTh2 --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 4   --stacks 1 --levels 5 --lr 1e-5  --batch_size 32  --dropout 0.5
run_scinet "ETTh2_RevIN_pl720"   --data ETTh2 --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 4   --stacks 1 --levels 5 --lr 1e-5  --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTh2_pl960"         --data ETTh2 --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1   --stacks 1 --levels 4 --lr 5e-5  --batch_size 128 --dropout 0.5
run_scinet "ETTh2_RevIN_pl960"   --data ETTh2 --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1   --stacks 1 --levels 4 --lr 5e-5  --batch_size 128 --dropout 0.5  --ours

# Table 1 — ETTm1
run_scinet "ETTm1_pl24"          --data ETTm1 --features M --seq_len 48  --label_len 24  --pred_len 24   --hidden-size 4 --stacks 1 --levels 3 --lr 5e-3   --batch_size 32  --dropout 0.5
run_scinet "ETTm1_RevIN_pl24"    --data ETTm1 --features M --seq_len 48  --label_len 24  --pred_len 24   --hidden-size 4 --stacks 1 --levels 3 --lr 5e-3   --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTm1_pl48"          --data ETTm1 --features M --seq_len 96  --label_len 48  --pred_len 48   --hidden-size 4 --stacks 2 --levels 4 --lr 1e-3   --batch_size 16  --dropout 0.5
run_scinet "ETTm1_RevIN_pl48"    --data ETTm1 --features M --seq_len 96  --label_len 48  --pred_len 48   --hidden-size 4 --stacks 2 --levels 4 --lr 1e-3   --batch_size 16  --dropout 0.5  --ours
run_scinet "ETTm1_pl96"          --data ETTm1 --features M --seq_len 384 --label_len 96  --pred_len 96   --hidden-size 0.5 --stacks 2 --levels 4 --lr 5e-5 --batch_size 32  --dropout 0.5
run_scinet "ETTm1_RevIN_pl96"    --data ETTm1 --features M --seq_len 384 --label_len 96  --pred_len 96   --hidden-size 0.5 --stacks 2 --levels 4 --lr 5e-5 --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTm1_pl288"         --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288  --hidden-size 4 --stacks 1 --levels 5 --lr 1e-5   --batch_size 32  --dropout 0.5
run_scinet "ETTm1_RevIN_pl288"   --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288  --hidden-size 4 --stacks 1 --levels 5 --lr 1e-5   --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTm1_pl672"         --data ETTm1 --features M --seq_len 672 --label_len 672 --pred_len 672  --hidden-size 4 --stacks 2 --levels 5 --lr 1e-5   --batch_size 32  --dropout 0.5
run_scinet "ETTm1_RevIN_pl672"   --data ETTm1 --features M --seq_len 672 --label_len 672 --pred_len 672  --hidden-size 4 --stacks 2 --levels 5 --lr 1e-5   --batch_size 32  --dropout 0.5  --ours
run_scinet "ETTm1_pl1344"        --data ETTm1 --features M --seq_len 672 --label_len 672 --pred_len 1344 --hidden-size 4 --stacks 2 --levels 5 --lr 1e-5   --batch_size 32  --dropout 0.5
run_scinet "ETTm1_RevIN_pl1344"  --data ETTm1 --features M --seq_len 672 --label_len 672 --pred_len 1344 --hidden-size 4 --stacks 2 --levels 5 --lr 1e-5   --batch_size 32  --dropout 0.5  --ours

# Table 1 — ECL
run_scinet "ECL_pl24"            --data ECL --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5
run_scinet "ECL_RevIN_pl24"      --data ECL --features M --seq_len 48  --label_len 24  --pred_len 24  --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  --ours
run_scinet "ECL_pl48"            --data ECL --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5
run_scinet "ECL_RevIN_pl48"      --data ECL --features M --seq_len 96  --label_len 48  --pred_len 48  --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  --ours
run_scinet "ECL_pl168"           --data ECL --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5
run_scinet "ECL_RevIN_pl168"     --data ECL --features M --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  --ours
run_scinet "ECL_pl336"           --data ECL --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5
run_scinet "ECL_RevIN_pl336"     --data ECL --features M --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  --ours
run_scinet "ECL_pl720"           --data ECL --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5
run_scinet "ECL_RevIN_pl720"     --data ECL --features M --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  --ours
run_scinet "ECL_pl960"           --data ECL --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5
run_scinet "ECL_RevIN_pl960"     --data ECL --features M --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  --ours

log "=== All done ==="
