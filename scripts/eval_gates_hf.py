#!/usr/bin/env python3
"""Barist.8 gates for the HF encoder shield (kuclab.hf_model.HFShield).
Same gate keys as eval_gates.py, plus heldout4 and per-category recall.
Writes reports/gates_hf.json.

Usage: python scripts/eval_gates_hf.py --shield checkpoints_b8 --tag Barist.8
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kuclab.hf_model import HFShield
from kuclab.langdetect import must_catch_hit
from eval_gates import USER_BENIGN, IDENTITY_BENIGN, IDENTITY_HATE, _gold_cat


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shield", default="checkpoints_b8")
    ap.add_argument("--tag", default="Barist.8")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sh = HFShield(args.shield, device=args.device)
    r = {}

    def det(rows, pf):
        texts = [x["text"] for x in rows]
        preds = [sh.predict(t) for t in texts]
        raw = [sh.raw(t)[0] >= sh.threshold for t in texts]  # true model-only (no regex)
        u = [i for i, x in enumerate(rows) if x["label"] == 1]
        b = [i for i, x in enumerate(rows) if x["label"] == 0]
        return (sum(1 for i in u if raw[i]), len(u),
                sum(1 for i in u if preds[i]["unsafe"]),
                sum(1 for i in b if preds[i]["unsafe"]), len(b))

    held2 = load(ROOT / "data/processed/mustcatch_heldout2.jsonl")
    d, n, dp, bfp, bn = det(held2, True)
    r["heldout2_model_only"] = f"{d}/{n}"
    r["heldout2_with_prefilter"] = f"{dp}/{n}"
    r["heldout2_benign_fp"] = f"{bfp}/{bn}"
    # per-category recall on heldout2 (pipeline-reported)
    h2u = [x for x in held2 if x["label"] == 1]
    for cat in ("sexual_minors", "self_harm"):
        gold = [i for i, x in enumerate(h2u) if (x.get("category") or _gold_cat(x["text"])) in (cat, {"sexual_minors": "sexual_abuse", "self_harm": "self_harm"}.get(cat, cat))]
        hit = sum(1 for i in gold if cat in sh.predict(h2u[i]["text"])["categories"])
        r[f"cat_recall_{cat}"] = round(hit / len(gold), 4) if gold else None

    held3 = load(ROOT / "data/processed/mustcatch_heldout3.jsonl")
    d3, n3, d3p, bfp3, bn3 = det(held3, True)
    r["heldout3_model_only"] = f"{d3}/{n3}"
    r["heldout3_with_prefilter"] = f"{d3p}/{n3}"

    h4p = ROOT / "data/processed/mustcatch_heldout4.jsonl"
    if h4p.exists():
        held4 = load(h4p)
        d4, n4, d4p, bfp4, bn4 = det(held4, True)
        r["heldout4_model_only"] = f"{d4}/{n4}"
        r["heldout4_with_prefilter"] = f"{d4p}/{n4}"
        r["heldout4_benign_fp"] = f"{bfp4}/{bn4}"

    csben = load(ROOT / "data/processed/cs_benign_eval.jsonl")
    preds = [sh.predict(x["text"]) for x in csben]
    r["cs_benign_fpr_raw"] = round(sum(1 for p, x in zip(preds, csben) if p["unsafe"]) / len(csben), 4)

    test = load(ROOT / "data/processed/test.jsonl")
    tp_ = [sh.predict(x["text"]) for x in test]
    y = [x["label"] for x in test]
    tp = sum(1 for l, p in zip(y, tp_) if l == 1 and p["unsafe"])
    fp = sum(1 for l, p in zip(y, tp_) if l == 0 and p["unsafe"])
    fn = sum(1 for l, p in zip(y, tp_) if l == 1 and not p["unsafe"])
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    r["test_f1"] = round(2 * pr * rc / max(1e-9, pr + rc), 4)
    r["test_precision"] = round(pr, 4)
    r["test_recall"] = round(rc, 4)

    ub = [sh.predict(t) for t in USER_BENIGN]
    r["user_benign_fp"] = f"{sum(1 for p in ub if p['unsafe'])}/{len(USER_BENIGN)}"
    idp = [sh.predict(t) for t in IDENTITY_BENIGN]
    r["identity_fp"] = round(sum(1 for p in idp if p["unsafe"]) / len(IDENTITY_BENIGN), 4)
    idh = [sh.predict(t) for t in IDENTITY_HATE]
    r["identity_recall"] = round(sum(1 for p in idh if p["unsafe"]) / len(IDENTITY_HATE), 4)

    # leak gate: no eval sentence may appear in train.jsonl (normalized)
    def _norm(t):
        return " ".join(t.lower().split())
    eval_norms = set()
    for p in ("data/processed/mustcatch_heldout2.jsonl", "data/processed/mustcatch_heldout3.jsonl",
              "data/processed/mustcatch_heldout4.jsonl", "data/processed/cs_benign_eval.jsonl",
              "data/regex_benign_test.jsonl", "data/processed/mustcatch_dev.jsonl"):
        q = ROOT / p
        if q.exists():
            for l in open(q, encoding="utf-8"):
                eval_norms.add(_norm(json.loads(l)["text"]))
    for t in USER_BENIGN + IDENTITY_BENIGN + IDENTITY_HATE:
        eval_norms.add(_norm(t))
    leaks = sum(1 for l in open(ROOT / "data/processed/train.jsonl", encoding="utf-8")
                if _norm(json.loads(l)["text"]) in eval_norms)
    r["train_eval_leak"] = leaks

    out = ROOT / "reports/gates_hf.json"
    out.write_text(json.dumps({"tag": args.tag, "shield": args.shield, "gates": r}, indent=2, ensure_ascii=False))
    for k, v in r.items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
