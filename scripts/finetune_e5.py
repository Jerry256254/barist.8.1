#!/usr/bin/env python3
"""Barist.8: fine-tune a pretrained multilingual encoder (intfloat/multilingual-e5-small)
with two heads: binary safe/unsafe (softmax) + multi-label categories (sigmoid, BCE masked).

Categories (OpenAI-like + grooming + jailbreak):
  0 sexual, 1 sexual_minors, 2 harassment, 3 harassment_threatening, 4 hate,
  5 hate_threatening, 6 self_harm, 7 self_harm_intent, 8 self_harm_instructions,
  9 violence, 10 violence_graphic, 11 illicit, 12 illicit_violent, 13 grooming, 14 jailbreak_injection

Old 10-cat indices in train.jsonl map via OLD2NEW.
Usage: python scripts/finetune_e5.py --out checkpoints_b8 [--epochs 2] [--base intfloat/multilingual-e5-small]
"""
import argparse, json, math, random, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CATEGORIES = ["sexual", "sexual_minors", "harassment", "harassment_threatening", "hate",
              "hate_threatening", "self_harm", "self_harm_intent", "self_harm_instructions",
              "violence", "violence_graphic", "illicit", "illicit_violent", "grooming",
              "jailbreak_injection"]
OLD_CATEGORIES = ["violence", "weapons_explosives", "drugs", "hate_harassment", "sexual_abuse",
                  "self_harm", "cybercrime_fraud", "jailbreak_injection", "illegal_marketplace", "other_unsafe"]
OLD2NEW = {"violence": "violence", "weapons_explosives": "illicit_violent", "drugs": "illicit",
           "hate_harassment": "hate", "sexual_abuse": "sexual_minors", "self_harm": "self_harm",
           "cybercrime_fraud": "illicit", "jailbreak_injection": "jailbreak_injection",
           "illegal_marketplace": "illicit", "other_unsafe": "illicit"}


class ShieldDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"][:4000], truncation=True, max_length=self.max_len)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["y_bin"] = torch.tensor(r["label"], dtype=torch.long)
        cats = torch.zeros(len(CATEGORIES))
        w = torch.ones(len(CATEGORIES))  # learn negatives on benign rows too
        for c in r.get("cats") or []:
            name = OLD_CATEGORIES[c] if isinstance(c, int) else c
            new = OLD2NEW.get(name, name)
            if new in CATEGORIES:
                cats[CATEGORIES.index(new)] = 1.0
        item["y_cats"] = cats
        item["w_cats"] = w
        return item


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ("input_ids", "attention_mask", "token_type_ids"):
            out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True,
                                                      padding_value=0)
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints_b8")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--init", default="")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.02)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModel.from_pretrained(args.base).to(device)
    base.gradient_checkpointing_enable()
    hidden = base.config.hidden_size
    head_bin = torch.nn.Sequential(torch.nn.Dropout(args.dropout), torch.nn.Linear(hidden, 2)).to(device)
    head_cats = torch.nn.Sequential(torch.nn.Dropout(args.dropout), torch.nn.Linear(hidden, len(CATEGORIES))).to(device)
    if args.init and (Path(args.init) / "heads.safetensors").exists():
        from safetensors.torch import load_file
        hh = load_file(str(Path(args.init) / "heads.safetensors"))
        (head_bin[-1] if isinstance(head_bin, torch.nn.Sequential) else head_bin).load_state_dict({"weight": hh["head_bin.weight"].to(device), "bias": hh["head_bin.bias"].to(device)})
        (head_cats[-1] if isinstance(head_cats, torch.nn.Sequential) else head_cats).load_state_dict({"weight": hh["head_cats.weight"].to(device), "bias": hh["head_cats.bias"].to(device)})
        print("loaded heads from", args.init, flush=True)

    rows = [json.loads(l) for l in open(ROOT / "data/processed/train.jsonl", encoding="utf-8")]
    random.shuffle(rows)
    nval = int(len(rows) * args.val_frac)
    val_rows, train_rows = rows[:nval], rows[nval:]
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    ds = ShieldDataset(train_rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=4)

    params = list(base.parameters()) + list(head_bin.parameters()) + list(head_cats.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    n_steps = math.ceil(len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.05,
                                                total_iters=n_steps)
    # class weights for binary head (unsafe is minority)
    npos = sum(r["label"] for r in train_rows)
    pos_weight = torch.tensor([1.0, args.pos_weight], device=device)
    print("pos_weight:", pos_weight.tolist(), flush=True)
    # per-category pos_weight so rare cats (sexual_minors, grooming, jailbreak) are not
    # drowned out by the majority 'hate'/'violence' labels (fixes the hate trash-bin)
    import collections
    cnt = collections.Counter()
    for r in train_rows:
        for c in r.get("cats") or []:
            name = OLD_CATEGORIES[c] if isinstance(c, int) else c
            cnt[OLD2NEW.get(name, name)] += 1
    n = len(train_rows)
    cat_pw = torch.ones(len(CATEGORIES))
    for i, c in enumerate(CATEGORIES):
        p = cnt.get(c, 0)
        if p > 0:
            cat_pw[i] = min(25.0, max(1.0, (n - p) / p * 0.25))
    cat_pw = cat_pw.to(device)
    print("cat pos_weight:", {c: round(v, 1) for c, v in zip(CATEGORIES, cat_pw.tolist())}, flush=True)

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    step = 0
    for epoch in range(int(args.epochs)):
        base.train(); head_bin.train(); head_cats.train()
        for batch in dl:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                h = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0]
                logits_bin = head_bin(h)
                loss_bin = torch.nn.functional.cross_entropy(logits_bin, batch["y_bin"], weight=pos_weight, label_smoothing=args.smooth)
                logits_cats = head_cats(h)
                loss_cat = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_cats, batch["y_cats"], weight=batch["w_cats"], pos_weight=cat_pw)
                loss = loss_bin + 0.5 * loss_cat
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            if step % 400 == 0:
                print(f"epoch {epoch} step {step}/{n_steps} loss={loss.item():.4f} bin={loss_bin.item():.4f} cat={loss_cat.item():.4f}", flush=True)
        # quick val
        base.eval(); head_bin.eval(); head_cats.eval()
        correct = tot = 0
        with torch.no_grad():
            for i in range(0, len(val_rows), 256):
                chunk = val_rows[i:i + 256]
                enc = tok([r["text"][:4000] for r in chunk], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to(device)
                h = base(**enc).last_hidden_state[:, 0]
                pred = head_bin(h).argmax(-1).cpu()
                correct += sum(int(p == r["label"]) for p, r in zip(pred, chunk))
                tot += len(chunk)
        print(f"epoch {epoch} val_acc={correct / max(1, tot):.4f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(out)          # fine-tuned encoder (HFShield loads it from here)
    tok.save_pretrained(out)
    from safetensors.torch import save_file
    hb = head_bin[-1] if isinstance(head_bin, torch.nn.Sequential) else head_bin
    hc = head_cats[-1] if isinstance(head_cats, torch.nn.Sequential) else head_cats
    save_file({"head_bin.weight": hb.weight.detach().cpu(),
               "head_bin.bias": hb.bias.detach().cpu(),
               "head_cats.weight": hc.weight.detach().cpu(),
               "head_cats.bias": hc.bias.detach().cpu()}, str(out / "heads.safetensors"))
    meta = {"base": args.base, "base_dir": ".", "categories": CATEGORIES, "max_len": args.max_len,
            "threshold": 0.5, "cat_threshold": 0.5, "name": "Barist.8",
            "category_block": False, "architecture": "e5-small + linear heads"}
    (out / "shield_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved to", out, flush=True)


if __name__ == "__main__":
    main()
