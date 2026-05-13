"""
Convert pretrained Qwen3-VL-4B-Instruct weights to the global-token architecture
and save a self-contained checkpoint.

Usage:
    python convert_weights.py [--src SRC_PATH] [--dst DST_PATH]

After conversion, load with:
    from qwen3vl_improved import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(DST_PATH, dtype="auto", device_map="auto")

No ignore_mismatched_sizes or adapt_weights_for_global_token needed.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; no GPU needed for conversion

import torch
from qwen3vl_improved import Qwen3VLForConditionalGeneration, adapt_weights_for_global_token

DEFAULT_SRC = "/mnt/nvme2/zys/models/Qwen3-VL-4B-Instruct"
DEFAULT_DST = "/mnt/nvme2/zys/models/Qwen3-VL-4B-Instruct-global-token"

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=DEFAULT_SRC)
parser.add_argument("--dst", default=DEFAULT_DST)
args = parser.parse_args()

print(f"[convert] src: {args.src}")
print(f"[convert] dst: {args.dst}")

# ── 1. Load pretrained weights into new architecture ───────────────────────────
print("[convert] loading model (CPU) ...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.src,
    dtype=torch.float16,
    device_map="cpu",
    ignore_mismatched_sizes=True,
)

# ── 2. Initialise global_proj from proj
adapt_weights_for_global_token(model, args.src)

# ── 2b. Reset fold_gates to float32 0.01 explicitly.
#        Scalar nn.Parameters can get silently corrupted when the model is cast
#        to float16 during from_pretrained — a tiny float16 value gets reinterpreted
#        incorrectly on reload.  Keeping them float32 avoids the round-trip issue.
with torch.no_grad():
    for gate in model.model.visual.fold_gates:
        gate.data = torch.tensor(0.01, dtype=torch.float32)
print(f"[convert] fold_gates: {[round(g.item(), 4) for g in model.model.visual.fold_gates]}")

# ── 3. Save new checkpoint ──────────────────────────────────────────────────────
os.makedirs(args.dst, exist_ok=True)
print(f"[convert] saving model to {args.dst} ...")
model.save_pretrained(args.dst, safe_serialization=True)

# ── 4. Copy processor/tokenizer files so the directory is self-contained ────────
SKIP_PREFIXES = ("model.safetensors",)
for fname in os.listdir(args.src):
    if any(fname.startswith(p) for p in SKIP_PREFIXES):
        continue
    src_f = os.path.join(args.src, fname)
    dst_f = os.path.join(args.dst, fname)
    if os.path.isfile(src_f) and not os.path.exists(dst_f):
        shutil.copy2(src_f, dst_f)
        print(f"[convert]   copied {fname}")

print("[convert] done.")
print(f"\nLoad with:\n  model = Qwen3VLForConditionalGeneration.from_pretrained('{args.dst}', dtype='auto', device_map='auto')")
