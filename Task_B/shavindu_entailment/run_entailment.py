"""
Part 2 checker for Shavindu.

DeBERTa-v3 is only run on sentences that BM25 already ranks as close to the fact.
The fact score is the highest entailment probability among those sentences.
A fact whose sentences all fall under the BM25 gate is labelled NS with no model call.

Probabilities are cached so the gate and the probability cut can be retuned
without running the model again.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Task_A" / "shavindu_bm25"))

from bm25_checker import BM25FactChecker, load_examples, tokenize  # noqa: E402

CACHE = Path(__file__).resolve().parent / "pair_probs.jsonl"
MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def sentences_of(text):
    import re
    out = []
    for chunk in re.split(r"</?s>", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            sent = re.sub(r"\s+", " ", sent).strip()
            if len(sent.split()) >= 3:
                out.append(sent)
    return out


def build_pairs(examples, checker):
    pairs = []
    for i, ex in enumerate(examples):
        query = checker.query_terms(ex["fact"], ex["passages"])
        seen = set()
        for passage in ex["passages"]:
            for sent in sentences_of(passage.get("text") or ""):
                if sent in seen:
                    continue
                seen.add(sent)
                doc = tokenize(sent, checker.stop, checker.stem)
                bm25 = checker.score_doc(query, doc) if query else 0.0
                pairs.append({
                    "fact_idx": i,
                    "premise": sent,
                    "bm25": bm25,
                })
    return pairs


def run_model(pairs, examples):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("device", device, "pairs", len(pairs), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(device)
    model.eval()
    batch_size = 8
    done = []
    with CACHE.open("w", encoding="utf-8") as f:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            premises = [p["premise"] for p in batch]
            hyps = [examples[p["fact_idx"]]["fact"] for p in batch]
            with torch.no_grad():
                inputs = tokenizer(premises, hyps, return_tensors="pt",
                                   truncation=True, max_length=256, padding=True)
                if device != "cpu":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row, prob in zip(batch, probs):
                rec = dict(row)
                rec["p_e"] = float(prob[0])
                rec["p_n"] = float(prob[1])
                rec["p_c"] = float(prob[2])
                f.write(json.dumps(rec) + "\n")
                done.append(rec)
            if start % 80 == 0:
                print(f"scored {start + len(batch)}/{len(pairs)}", flush=True)
            del inputs, logits, probs
    return done


def load_cache():
    rows = []
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def fact_score(rows_for_fact, mode, gate, topk):
    kept = [r for r in rows_for_fact if r["bm25"] >= gate]
    if mode == "best_bm25":
        if not rows_for_fact:
            return None
        best = max(rows_for_fact, key=lambda r: r["bm25"])
        if best["bm25"] < gate:
            return None
        return best
    kept.sort(key=lambda r: r["bm25"], reverse=True)
    kept = kept[:topk]
    return kept


def evaluate(rows, examples, mode, gate, topk, theta, rule):
    by_fact = {}
    for row in rows:
        by_fact.setdefault(row["fact_idx"], []).append(row)
    tp = fp = fn = tn = 0
    n_calls = 0
    for i, ex in enumerate(examples):
        gold = ex["label"]
        chosen = fact_score(by_fact.get(i, []), mode, gate, topk)
        if chosen is None or chosen == []:
            pred = "NS"
        else:
            if mode == "best_bm25":
                used = [chosen]
            else:
                used = chosen
            n_calls += len(used)
            if rule == "max_pe":
                score = max(r["p_e"] for r in used)
                pred = "S" if score >= theta else "NS"
            elif rule == "max_margin":
                score = max(r["p_e"] - r["p_c"] for r in used)
                pred = "S" if score >= theta else "NS"
            else:
                pred = "S" if any(r["p_e"] >= r["p_n"] and r["p_e"] >= r["p_c"] for r in used) else "NS"
        if pred == "S" and gold == "S":
            tp += 1
        elif pred == "S" and gold == "NS":
            fp += 1
        elif pred == "NS" and gold == "S":
            fn += 1
        else:
            tn += 1
    correct = tp + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(correct=correct, acc=correct / len(examples), precision=prec, recall=rec,
                f1=f1, tp=tp, fp=fp, fn=fn, tn=tn, calls=n_calls)


def sweep(rows, examples):
    print("\nmode gate topk rule theta correct acc P R F1 calls")
    best = None
    configs = []
    for mode, topk in (("topk", 1), ("topk", 3), ("topk", 5), ("best_bm25", 1), ("topk", 99)):
        for gate in (0.0, 0.05, 0.10, 0.15, 0.20):
            for rule, thetas in (
                ("max_pe", [i / 100 for i in range(5, 80, 5)]),
                ("max_margin", [i / 100 for i in range(0, 60, 5)]),
                ("discrete", [None]),
            ):
                for theta in thetas:
                    result = evaluate(rows, examples, mode, gate, topk, theta if theta is not None else 0.0, rule)
                    result.update(mode=mode, gate=gate, topk=topk, rule=rule, theta=theta)
                    configs.append(result)
                    key = (result["acc"], result["f1"], -result["calls"])
                    if best is None or key > best[0]:
                        best = (key, result)
    # print a short list: best per mode, plus the overall best
    configs.sort(key=lambda r: (r["acc"], r["f1"]), reverse=True)
    shown = 0
    for r in configs:
        if shown >= 12:
            break
        print(
            f"{r['mode']:10} gate={r['gate']:.2f} k={r['topk']:<2} {r['rule']:10} "
            f"th={r['theta']}  {r['correct']}/221={r['acc']:.3f}  "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} calls={r['calls']}"
        )
        shown += 1
    return configs


def main():
    examples = load_examples()
    checker = BM25FactChecker(k1=1.2, b=0.25, unit="sentence", threshold=0.197)
    if CACHE.exists() and CACHE.stat().st_size > 0:
        print("loading cache", CACHE)
        rows = load_cache()
    else:
        pairs = build_pairs(examples, checker)
        print("unique sentences", len(pairs), flush=True)
        rows = run_model(pairs, examples)
    sweep(rows, examples)


if __name__ == "__main__":
    main()
