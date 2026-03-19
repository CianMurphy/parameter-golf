#!/usr/bin/env bash
# === RunPod Setup & Training Script ===
# Paste this entire script into your RunPod 8xH100 terminal.
# It will: install deps, download data, run 3 seeds, and print results.
set -euo pipefail

cd /workspace

# 1. Clone YOUR fork (with the submission branch)
git clone --branch submission/normuon-ema-cosine https://github.com/CianMurphy/parameter-golf.git
cd parameter-golf

# 2. Install dependencies
pip install -q sentencepiece zstandard numpy

# 3. Download the SP-1024 FineWeb dataset (full 80 shards)
python3 data/cached_challenge_fineweb.py --variant sp1024

# 4. Run seed 1337 (canonical run)
echo ""
echo "=========================================="
echo "=== SEED 1337 (canonical run) ==="
echo "=========================================="
RUN_ID=normuon_ema_cosine_seed1337 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_gpt.py

# Copy logs
cp logs/normuon_ema_cosine_seed1337.txt \
   records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train.log

# 5. Run seed 1338
echo ""
echo "=========================================="
echo "=== SEED 1338 ==="
echo "=========================================="
RUN_ID=normuon_ema_cosine_seed1338 \
SEED=1338 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_gpt.py

cp logs/normuon_ema_cosine_seed1338.txt \
   records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_seed1338.log

# 6. Run seed 1339
echo ""
echo "=========================================="
echo "=== SEED 1339 ==="
echo "=========================================="
RUN_ID=normuon_ema_cosine_seed1339 \
SEED=1339 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_gpt.py

cp logs/normuon_ema_cosine_seed1339.txt \
   records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_seed1339.log

# 7. Print all final scores
echo ""
echo "=========================================="
echo "=== FINAL RESULTS ==="
echo "=========================================="
echo "Seed 1337:"
grep "final_int6_zstd_roundtrip_exact" records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train.log
echo "Seed 1338:"
grep "final_int6_zstd_roundtrip_exact" records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_seed1338.log
echo "Seed 1339:"
grep "final_int6_zstd_roundtrip_exact" records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train_seed1339.log
echo ""
echo "Total submission size:"
grep "Total submission size" records/track_10min_16mb/2026-03-19_NorMuon_EMA_CosineWarmdown/train.log
echo ""
echo "=== DONE. Copy the scores above and update submission.json ==="
