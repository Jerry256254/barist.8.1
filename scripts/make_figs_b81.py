#!/usr/bin/env python3
"""Barist.8.1 performance charts -> reports/figs/*.png

    .venv/bin/python scripts/make_figs_b81.py
Sources: reports/gates.json (Barist.7), reports/gates_hf.json (Barist.8.1),
reports/benchmark_cpu.json (Barist.7 latency), reports/bench_b81.json +
ONNX export runs (Barist.8.1 latency).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

g81 = json.load(open(ROOT / "reports/gates_hf.json"))["gates"]
g7 = json.load(open(ROOT / "reports/gates.json"))
b7 = {x["gate"]: x["Barist.7"] for x in g7["gates"]}

def frac(s):
    a, b = s.split("/")
    return int(a) / int(b)

# ---- 1. gates: recall/F1 (higher better) + FPR (lower better) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
rec_labels = ["heldout2\n(pipeline)", "heldout3\n(pipeline)", "heldout4\n(pipeline)", "test F1", "identity\nrecall", "cat_recall\nsexual_minors"]
rec7 = [frac(b7["heldout2_with_prefilter"]), frac(b7["heldout3_with_prefilter"]), np.nan,
        0.822, 0.625, 0.75]
rec81 = [frac(g81["heldout2_with_prefilter"]), frac(g81["heldout3_with_prefilter"]),
         frac(g81["heldout4_with_prefilter"]), g81["test_f1"], g81["identity_recall"],
         g81["cat_recall_sexual_minors"]]
x = np.arange(len(rec_labels)); w = 0.38
axes[0].bar(x - w/2, [0 if np.isnan(v) else v for v in rec7], w, label="Barist.7", color="#888")
axes[0].bar(x + w/2, rec81, w, label="Barist.8.1", color="#2a7")
axes[0].set_xticks(x); axes[0].set_xticklabels(rec_labels, fontsize=8)
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("recall / F1")
axes[0].set_title("Detection quality (higher = better)")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(rec81):
    axes[0].text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

fp_labels = ["heldout2\nbenign FP", "cs_benign\nFPR", "user benign\nFP", "identity\nFP"]
fp7 = [frac(b7["heldout2_benign_fp"]), b7["cs_benign_fpr_raw"], frac(b7["user_benign_fp"]), b7["identity_fp"]]
fp81 = [frac(g81["heldout2_benign_fp"]), g81["cs_benign_fpr_raw"], frac(g81["user_benign_fp"]), g81["identity_fp"]]
axes[1].bar(x[:4] - w/2, fp7, w, label="Barist.7", color="#888")
axes[1].bar(x[:4] + w/2, fp81, w, label="Barist.8.1", color="#2a7")
axes[1].set_xticks(x[:4]); axes[1].set_xticklabels(fp_labels, fontsize=8)
axes[1].set_ylabel("false-positive rate"); axes[1].set_title("False positives (lower = better)")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(fp81):
    axes[1].text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
fig.suptitle("Barist.7 vs Barist.8.1 — release gates (reports/gates.json, reports/gates_hf.json)")
fig.tight_layout()
fig.savefig(FIGS / "gates_b7_vs_b81.png", dpi=150)
plt.close(fig)

# ---- 2. latency ----
fig, ax = plt.subplots(figsize=(9, 4.4))
names = ["Barist.7 (903K)\ntorch CPU", "Barist.8.1 (118M)\nONNX CPU dev", "Barist.8.1\nONNX CPU (reviewer)", "Network moderation API\n(e.g. OpenAI, est.)"]
vals = [0.71, 5.4, 40.0, 200.0]
colors = ["#888", "#2a7", "#2a7", "#c66"]
bars = ax.bar(names, vals, color=colors)
ax.set_yscale("log"); ax.set_ylabel("ms / text (log scale)")
ax.set_title("Latency per text — local & offline vs network API")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v * 1.15, f"{v:g} ms", ha="center", fontsize=10)
ax.text(0.01, 0.95, "Barist.8.1 dev CPU: 5.4–8.5 ms across runs (reports/bench_b81.json); reviewer hardware ~40 ms;\nnetwork API ~100–300 ms round-trip (estimate, no key in this environment)",
        transform=ax.transAxes, fontsize=8, va="top")
fig.tight_layout()
fig.savefig(FIGS / "latency_b81.png", dpi=150)
plt.close(fig)

# ---- 3. size / params ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["Barist.7", "Barist.8.1"], [0.9, 118], color=["#888", "#2a7"])
axes[0].set_yscale("log"); axes[0].set_ylabel("parameters (M, log)")
axes[0].set_title("Model size (params)")
for i, v in enumerate([0.9, 118]):
    axes[0].text(i, v * 1.2, f"{v:g}M", ha="center")
axes[1].bar(["Barist.7", "Barist.8.1 fp32", "Barist.8.1 ONNX"], [3.45, 449, 470], color=["#888", "#2a7", "#7ad"])
axes[1].set_yscale("log"); axes[1].set_ylabel("checkpoint (MB, log)")
axes[1].set_title("Checkpoint size on disk")
for i, v in enumerate([3.45, 449, 470]):
    axes[1].text(i, v * 1.15, f"{v:g} MB", ha="center")
fig.suptitle("Size trade-off: flagship accuracy vs the old 903K toy")
fig.tight_layout()
fig.savefig(FIGS / "size_params_b81.png", dpi=150)
plt.close(fig)

# ---- 4. category head recall ----
fig, ax = plt.subplots(figsize=(7, 4))
cats = ["sexual_minors", "self_harm"]
v7 = [0.75, 0.5714]
v81 = [g81["cat_recall_sexual_minors"], g81["cat_recall_self_harm"]]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, v7, w, label="Barist.8", color="#888")
ax.bar(x + w/2, v81, w, label="Barist.8.1", color="#2a7")
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0, 1.05)
ax.set_ylabel("recall on heldout2 (pipeline-reported category)")
ax.set_title("Category head — per-category pos_weight fixed the hate trash-bin")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(v81):
    ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center")
fig.tight_layout()
fig.savefig(FIGS / "cat_recall_b81.png", dpi=150)
plt.close(fig)

print("figs written:", [p.name for p in sorted(FIGS.glob("*b81*.png"))])
