"""
BM25 fact checker (Shavindu).

The fact is the query. Each Wikipedia sentence is a document.
Score = max normalised BM25 over sentences. Predict S if the score is at least theta.

Normalisation divides by the sum of query IDF weights, so the score is in [0, 1]
and the same 0.001 threshold grid as the other Part 1 checkers can be used.

IDF is computed on the unique passages in the retrieval file. Labels are not read
for that step. The dev set selected sentences, k1 = 1.2, b = 0.25, theta = 0.197
(191/221).
"""

import json
import re
from collections import Counter
from math import log
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "data" / "dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = ROOT / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"

# Function words. Negations are kept on purpose.
STOP = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "also", "onto", "per", "via", "within", "without", "across", "among",
    "however", "therefore", "thus",
}

STEMMER = PorterStemmer()


def tokenize(text, stop=True, stem=True):
    text = (text or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    out = []
    for tok in text.split():
        if stop and tok in STOP:
            continue
        if stem:
            tok = STEMMER.stem(tok)
        if tok:
            out.append(tok)
    return out


def title_tokens(title, stem=True):
    title = re.sub(r"\([^)]*\)", " ", title or "")
    return set(tokenize(title, stop=False, stem=stem))


class BM25FactChecker:
    def __init__(self, k1=1.2, b=0.25, threshold=0.197, drop_title=True,
                 stop=True, stem=True, unit="sentence",
                 passages_path=PASSAGES_PATH):
        self.k1 = k1
        self.b = b
        self.threshold = threshold
        self.drop_title = drop_title
        self.stop = stop
        self.stem = stem
        self.unit = unit
        self.idf, self.n_docs, self.avgdl = self._fit(passages_path)

    def _fit(self, path):
        unique = {}
        lengths = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                for passage in json.loads(line)["passages"]:
                    key = (passage.get("title"), passage.get("text"))
                    if key in unique:
                        continue
                    if self.unit == "sentence":
                        docs = []
                        for chunk in re.split(r"</?s>", passage.get("text") or ""):
                            chunk = chunk.strip()
                            if not chunk:
                                continue
                            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                                toks = tokenize(sent, self.stop, self.stem)
                                if toks:
                                    docs.append(toks)
                        unique[key] = docs
                        lengths.extend(len(d) for d in docs)
                    else:
                        toks = tokenize(passage.get("text") or "", self.stop, self.stem)
                        unique[key] = [toks] if toks else []
                        if toks:
                            lengths.append(len(toks))
        df = Counter()
        n_docs = 0
        for docs in unique.values():
            for doc in docs:
                df.update(set(doc))
                n_docs += 1
        # Passage-level IDF even for sentence scoring: a term's rarity is
        # counted on passages, which is the retrieval unit in the file.
        # Recompute passage df always, so sentence mode does not inflate df.
        if self.unit == "sentence":
            df = Counter()
            n_docs = 0
            seen = set()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    for passage in json.loads(line)["passages"]:
                        key = (passage.get("title"), passage.get("text"))
                        if key in seen:
                            continue
                        seen.add(key)
                        toks = tokenize(passage.get("text") or "", self.stop, self.stem)
                        if toks:
                            df.update(set(toks))
                            n_docs += 1
        idf = {t: log((n_docs - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
        avgdl = float(np.mean(lengths)) if lengths else 1.0
        self._n_passages = n_docs
        return idf, n_docs, avgdl

    def _docs(self, passages):
        docs = []
        for passage in passages:
            text = passage.get("text") or ""
            if self.unit == "passage":
                toks = tokenize(text, self.stop, self.stem)
                if toks:
                    docs.append(toks)
                continue
            for chunk in re.split(r"</?s>", text):
                chunk = chunk.strip()
                if not chunk:
                    continue
                for sent in re.split(r"(?<=[.!?])\s+", chunk):
                    toks = tokenize(sent, self.stop, self.stem)
                    if toks:
                        docs.append(toks)
        return docs

    def score_doc(self, query, doc):
        """Normalised BM25 of one already-tokenised document. query and doc are token lists."""
        if not query or not doc:
            return 0.0
        unseen = log((self.n_docs + 0.5) / 0.5 + 1.0)
        weights = [self.idf.get(t, unseen) for t in query]
        weight_sum = sum(weights) or 1.0
        tf = Counter(doc)
        norm = 1.0 - self.b + self.b * (len(doc) / self.avgdl)
        raw = 0.0
        for term, weight in zip(query, weights):
            freq = tf[term]
            if freq == 0:
                continue
            raw += weight * (freq / (freq + self.k1 * norm))
        return raw / weight_sum

    def query_terms(self, fact, passages):
        title = passages[0].get("title") if passages else ""
        query = tokenize(fact, self.stop, self.stem)
        if self.drop_title:
            kept = [t for t in query if t not in title_tokens(title, self.stem)]
            if kept:
                query = kept
        return query

    def score(self, fact, passages):
        query = self.query_terms(fact, passages)
        if not query:
            return 0.0
        best = 0.0
        for doc in self._docs(passages):
            value = self.score_doc(query, doc)
            if value > best:
                best = value
        return best

    def predict(self, fact, passages):
        return "S" if self.score(fact, passages) >= self.threshold else "NS"


def load_examples():
    fact_to_passages = {}
    with PASSAGES_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fact_to_passages[row["sent"]] = row["passages"]
    examples = []
    with LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("annotations"):
                continue
            for sent in row["annotations"]:
                for fact in sent.get("human-atomic-facts") or []:
                    text = fact["text"]
                    if text not in fact_to_passages:
                        continue
                    label = "NS" if fact["label"] == "IR" else fact["label"]
                    examples.append({
                        "fact": text,
                        "passages": fact_to_passages[text],
                        "label": label,
                        "raw_label": fact["label"],
                        "topic": row.get("topic", ""),
                    })
    return examples


def best_threshold(scores, y):
    best = None
    for step in range(1001):
        theta = step / 1000.0
        pred = scores >= theta
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        correct = tp + tn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = correct / len(y)
        key = (acc, f1, -theta)
        if best is None or key > best[0]:
            best = (key, theta, acc, prec, rec, f1, tp, fp, fn, tn)
    _, theta, acc, prec, rec, f1, tp, fp, fn, tn = best
    return dict(threshold=theta, accuracy=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def cv_accuracy(scores, y, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    accs = []
    for test_idx in np.array_split(indices, 5):
        mask = np.ones(len(y), dtype=bool)
        mask[test_idx] = False
        theta = best_threshold(scores[mask], y[mask])["threshold"]
        pred = scores[test_idx] >= theta
        accs.append(float((pred == y[test_idx]).mean()))
    return float(np.mean(accs))


def grouped_cv(scores, y, topics, seed=42):
    rng = np.random.default_rng(seed)
    names = np.array(sorted(set(topics)))
    rng.shuffle(names)
    accs = []
    topics = np.asarray(topics)
    for held in np.array_split(names, 5):
        test = np.isin(topics, held)
        if test.all() or not test.any():
            continue
        theta = best_threshold(scores[~test], y[~test])["threshold"]
        pred = scores[test] >= theta
        accs.append(float((pred == y[test]).mean()))
    return float(np.mean(accs))


def evaluate(checker, examples):
    y = np.array([1 if ex["label"] == "S" else 0 for ex in examples])
    scores = np.array([checker.score(ex["fact"], ex["passages"]) for ex in examples])
    # Retune so a grid row is not stuck with the default threshold.
    tuned = best_threshold(scores, y)
    topics = [ex["topic"] for ex in examples]
    tuned["cv"] = cv_accuracy(scores, y)
    tuned["grouped_cv"] = grouped_cv(scores, y, topics)
    return scores, tuned


def print_row(name, tuned):
    correct = tuned["tp"] + tuned["tn"]
    print(
        f"{name:42} th={tuned['threshold']:.3f}  {correct}/221={tuned['accuracy']:.4f}  "
        f"cv={tuned['cv']:.4f}  gcv={tuned['grouped_cv']:.4f}  "
        f"P={tuned['precision']:.3f} R={tuned['recall']:.3f} F1={tuned['f1']:.3f}  "
        f"tp={tuned['tp']} fp={tuned['fp']} fn={tuned['fn']} tn={tuned['tn']}"
    )


if __name__ == "__main__":
    examples = load_examples()
    print(f"Loaded {len(examples)} facts")
    configs = []
    for unit in ("passage", "sentence"):
        for b in (0.0, 0.25, 0.75, 1.0):
            configs.append((f"k1=1.2 b={b} {unit}", dict(k1=1.2, b=b, unit=unit)))
    for k1 in (0.6, 0.9, 1.5, 2.0):
        configs.append((f"k1={k1} b=0 passage", dict(k1=k1, b=0.0, unit="passage")))
    configs.append(("no title", dict(k1=1.2, b=0.0, unit="passage", drop_title=False)))
    configs.append(("no stem", dict(k1=1.2, b=0.0, unit="passage", stem=False)))
    configs.append(("no stop", dict(k1=1.2, b=0.0, unit="passage", stop=False)))

    for name, kwargs in configs:
        checker = BM25FactChecker(**kwargs)
        _, tuned = evaluate(checker, examples)
        print_row(name, tuned)
