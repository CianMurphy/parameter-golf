# NorMuon + EMA + Cosine Warmdown

Three novel improvements over the current best submission (ArjunAutoResearch, 1.165 bpb):

## What's new

**1. NorMuon optimizer** (replaces standard Muon)
From [NorMuon paper](https://github.com/zichongli5/NorMuon). Adds per-row adaptive step sizes via second moment estimation on top of Muon's Newton-Schulz orthogonalization. Each row of the update matrix gets its own adaptive learning rate, similar to how Adam provides per-parameter step sizes. Uses `all_gather` for distributed communication instead of `all_reduce` on flat update buffers.

**2. EMA (Exponential Moving Average) of weights**
Maintains a running average of model parameters during training (decay=0.999, starting at step 500). The EMA weights are used for the final model instead of the last-step weights. This smooths out optimizer noise and typically improves generalization by 0.005-0.01 bpb. Zero artifact size cost since only the EMA weights are serialized.

**3. Cosine warmdown schedule** (replaces linear)
Uses `0.5 * (1 + cos(pi * progress))` during the warmdown phase instead of linear decay. Cosine schedule holds higher LR for longer at the start of warmdown, then drops quickly at the end. This allows the model to continue learning more aggressively before final convergence.

## Inherited techniques (from prior submissions)

- **Wider MLP** (`MLP_MULT=3.0`, hidden=1536) from PR #70
- **Longer training context** (`TRAIN_SEQ_LEN=4096`) from PR #52
- **Optimizer tuning**: `MATRIX_LR=SCALAR_LR=0.02`, `TIED_EMBED_LR=0.03`, `MUON_MOMENTUM=0.99`, `WARMDOWN_ITERS=3000`
- **Smaller batch** (`TRAIN_BATCH_TOKENS=393216`) for more updates per wallclock
- **int6 per-row quantization** on MLP+attention weights from PR #70
- **fp16 tied embedding passthrough** from PR #42
- **zstd-22 compression** for better artifact compression
- **Sliding window evaluation** (stride=64, seq_len=4096) from PR #50

## Config

All hyperparameters are baked into the script as defaults. No env var overrides needed.

```
# Model
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=3.0  TIE_EMBEDDINGS=1

# Training
TRAIN_SEQ_LEN=4096  TRAIN_BATCH_TOKENS=393216  MAX_WALLCLOCK_SECONDS=600

# Optimizer (NorMuon)
MATRIX_LR=0.02  SCALAR_LR=0.02  TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99  MUON_BETA2=0.95
MUON_MOMENTUM_WARMUP_STEPS=1500  MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000 (cosine schedule)

# EMA
EMA_DECAY=0.999  EMA_START_STEP=500

# Eval (sliding window)
EVAL_STRIDE=64  EVAL_BATCH_SEQS=16
```

## How to reproduce

1. Clone the repo and download the SP-1024 dataset:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
pip install zstandard  # for better compression
```

2. Run training + evaluation:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training stops at the 10-minute wallclock cap. Sliding window evaluation runs automatically afterward (~3 min). The final `final_int6_zstd_roundtrip_exact` line in the log is the submission score.

To run with a different seed: `SEED=1338 torchrun ...`

## Expected results

Based on component analysis:
- ArjunAutoResearch baseline: **1.165 bpb**
- NorMuon optimizer: est. **-0.003 to -0.005 bpb** (adaptive per-row step sizes improve convergence)
- EMA weights: est. **-0.005 to -0.010 bpb** (smooths optimizer noise in final model)
- Cosine warmdown: est. **-0.002 to -0.003 bpb** (better LR schedule)
- **Expected total: ~1.148-1.155 bpb**

## Files

- `train_gpt.py`: the script, with all settings baked in as defaults
- `submission.json`: leaderboard metadata (scores filled after running)
- `README.md`: this file
