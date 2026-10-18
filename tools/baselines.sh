#!/bin/bash
# Equal-time baselines on FFHQ-256 (docs/results.md). Each run gets the same wall-clock budget on the same GPU;
# the last snapshot before the budget ends is evaluated with the same code as nightgan (`nightgan eval --images`).
#
#   tools/baselines.sh stylegan2|stylegan2-resume|r3gan|r3gan-resume|lightweight-gan|lightweight-gan-resume|eval-all
#
# Paths (override with environment variables):
#   FFHQ_IMAGES  folder of the 70k FFHQ 256x256 PNGs (tools/parquet_to_images.py)
#   FFHQ_ZIP     the same images as a StyleGAN dataset zip (stylegan3/dataset_tool.py --source FFHQ_IMAGES --dest ...)
#   FFHQ_DATA    nightgan dataset folder (nightgan prepare), for the reference statistics
#   SG3          the modernized StyleGAN3 fork (github.com/AlexMelanFromRingo/stylegan3), for StyleGAN2-ADA
#   R3GAN        R3GAN (github.com/brownvc/R3GAN) with torch_utils/custom_ops.py, misc.py, ops/*_gradfix.py and
#                dnnlib/util.py taken from the fork (current PyTorch / Python), its own bf16 kernels kept, and
#                float Adam betas in train.py / training/training_loop.py (PyTorch rejects betas=[0, 0])
#   LWGAN        the lightweight_gan command of an environment with lightweight-gan 1.3.0
#                (github.com/AlexMelanFromRingo/lightweight-gan-cpu)
#   SG3_PY       a Python environment with the fork's requirements.txt
set -euo pipefail
BUDGET=${BUDGET:-3h}
OUT=${OUT:-$HOME/runs/baselines}
FFHQ_IMAGES=${FFHQ_IMAGES:-$HOME/datasets/ffhq256}
FFHQ_ZIP=${FFHQ_ZIP:-$HOME/datasets/ffhq256-sg.zip}
FFHQ_DATA=${FFHQ_DATA:-$HOME/datasets/nightgan/ffhq256}
SG3=${SG3:-$HOME/Python/stylegan3}
R3GAN=${R3GAN:-$HOME/Python/baselines/R3GAN}
SG3_PY=${SG3_PY:?python of the StyleGAN3 environment}
LWGAN=${LWGAN:?lightweight_gan command}
NUM=${NUM:-10000}
mkdir -p "$OUT"
if ! timeout "$BUDGET" true 2>/dev/null; then  # e.g. "1h36m" is not a valid duration for timeout(1): use "96m"
    echo "invalid BUDGET '$BUDGET' (use one unit: 3h, 96m, 5400s)" >&2; exit 2
fi
budget() {  # run "$@" for at most BUDGET; 124 (the budget ran out) is the expected end, other failures are errors
    local rc=0
    timeout "$BUDGET" "$@" || rc=$?
    if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then echo "$1 failed with exit code $rc" >&2; exit $rc; fi
}

case "$1" in
stylegan2)  # StyleGAN2-ADA, FFHQ 256 settings of the StyleGAN2-ADA paper (gamma 1), batch 32 in 4 passes of 8
    cd "$SG3"
    budget "$SG3_PY" train.py --outdir="$OUT/stylegan2" --cfg=stylegan2 --data="$FFHQ_ZIP" --gpus=1 \
        --batch=32 --batch-gpu=8 --gamma=1 --mirror=1 --aug=ada --kimg=100000 --snap=2 --metrics=none \
        > "$OUT/stylegan2.log" 2>&1 ;;
stylegan2-resume)  # continue an interrupted StyleGAN2 run from its last snapshot for the rest of the budget
    pkl=$(ls -t "$OUT"/stylegan2/*/network-snapshot-*.pkl | head -1)
    cd "$SG3"
    budget "$SG3_PY" train.py --outdir="$OUT/stylegan2" --cfg=stylegan2 --data="$FFHQ_ZIP" --gpus=1 \
        --batch=32 --batch-gpu=8 --gamma=1 --mirror=1 --aug=ada --kimg=100000 --snap=2 --metrics=none \
        --resume="$pkl" > "$OUT/stylegan2-resume.log" 2>&1 ;;
r3gan)      # R3GAN's FFHQ-256 preset, batch 64 (the paper uses 256 on 8 GPUs) in passes of 8
    cd "$R3GAN"
    budget "$SG3_PY" train.py --outdir="$OUT/r3gan" --data="$FFHQ_ZIP" --gpus=1 --batch=64 \
        --g-batch-gpu=8 --d-batch-gpu=8 --mirror=1 --aug=1 --preset=FFHQ-256 --tick=1 --snap=10 --metrics=none \
        > "$OUT/r3gan.log" 2>&1 ;;
r3gan-resume)  # continue an interrupted R3GAN run from its last snapshot (snapshots every 2 kimg, ~9 min)
    pkl=$(ls -t "$OUT"/r3gan/*/network-snapshot-*.pkl | head -1)
    cd "$R3GAN"
    budget "$SG3_PY" train.py --outdir="$OUT/r3gan" --data="$FFHQ_ZIP" --gpus=1 --batch=64 \
        --g-batch-gpu=8 --d-batch-gpu=8 --mirror=1 --aug=1 --preset=FFHQ-256 --tick=1 --snap=2 --metrics=none \
        --resume="$pkl" > "$OUT/r3gan-resume.log" 2>&1 ;;
lightweight-gan-resume)  # lightweight-gan continues from its latest checkpoint by itself; save more often near the end
    budget "$LWGAN" --data "$FFHQ_IMAGES" --name ffhq256 --image-size 256 \
        --models-dir "$OUT/lwgan/models" --results-dir "$OUT/lwgan/results" --amp --num-train-steps 10000000 \
        --save-every 250 > "$OUT/lightweight-gan-resume.log" 2>&1 ;;
lightweight-gan)  # lightweight-gan defaults (batch 10 x 4 accumulation), bf16
    budget "$LWGAN" --data "$FFHQ_IMAGES" --name ffhq256 --image-size 256 \
        --models-dir "$OUT/lwgan/models" --results-dir "$OUT/lwgan/results" --amp --num-train-steps 10000000 \
        --save-every 500 > "$OUT/lightweight-gan.log" 2>&1 ;;
eval-all)   # NUM images of the last snapshot of each, scored like nightgan
    sg2=$(ls -t "$OUT"/stylegan2/*/network-snapshot-*.pkl | head -1)
    (cd "$SG3" && "$SG3_PY" gen_images.py --network="$sg2" --seeds=0-$((NUM - 1)) --trunc=1 --outdir="$OUT/eval/stylegan2")
    r3=$(ls -t "$OUT"/r3gan/*/network-snapshot-*.pkl | head -1)
    (cd "$R3GAN" && "$SG3_PY" gen_images.py --network="$r3" --seeds=0-$((NUM - 1)) --outdir="$OUT/eval/r3gan")
    for i in $(seq 0 $((NUM / 2000 - 1))); do
        "$LWGAN" --name ffhq256 --models-dir "$OUT/lwgan/models" \
            --results-dir "$OUT/eval/lwgan-$i" --generate --num-image-tiles 2000 --generate-types "['ema']"
    done
    for d in stylegan2 r3gan; do
        nightgan eval "$FFHQ_DATA" --images "$OUT/eval/$d" --metrics fid,kid,fd_dinov2 | tee "$OUT/eval/$d.txt"
    done
    mkdir -p "$OUT/eval/lwgan" && find "$OUT"/eval/lwgan-* -name '*-ema.*' -exec mv {} "$OUT/eval/lwgan/" \;
    nightgan eval "$FFHQ_DATA" --images "$OUT/eval/lwgan" --metrics fid,kid,fd_dinov2 | tee "$OUT/eval/lwgan.txt" ;;
*) echo "usage: $0 stylegan2|stylegan2-resume|r3gan|r3gan-resume|lightweight-gan|lightweight-gan-resume|eval-all"; exit 1 ;;
esac
