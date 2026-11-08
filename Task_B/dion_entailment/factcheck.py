import gc
import html
import re
import unicodedata
from collections import Counter
from typing import List, Sequence

from nltk.stem import PorterStemmer

_STOPWORDS = frozenset(
    "a an the of to and in for on with at by from as is was were be been being "
    "it this that these those he she they them his her their its or but if "
    "who whom which what when where into over after before about between "
    "than then so such too very can could would should may might must will just "
    "also only own same other onto per via am do does did has have had".split()
)
_STEMMER = PorterStemmer()
_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[^\W_]+")
_WINDOW = 8
_STRIDE = 4
# Cut that maximises accuracy for maximum ROUGE-1 recall on the development set.
_ROUGE1_RECALL_THRESHOLD = 0.42857142857142855
# Part 2. Maximum entailment probability over sentence, pair, and passage
# evidence that already has ROUGE-1 recall of at least 0.30. Tuned on the
# development set: 0.25 is the highest cut that still scores 186/221.
_ENTAILMENT_THRESHOLD = 0.25
_PRUNE_OVERLAP = 0.30
_MAX_PREMISES = 48
_EVIDENCE_KINDS = ("sentence", "pair", "passage")
_DECISION = "max_prob"
_NLI_BATCH = 8
_SENTENCE = re.compile(r"<s>(.*?)</s>", re.S)
_BOILERPLATE = frozenset({"references.", "see also.", "external links.", "notes."})


def _tokenize(text: str) -> List[str]:
    """Clean one string into stemmed content words. text: fact, passage, or title. Returns stems."""
    cleaned = unicodedata.normalize("NFKC", html.unescape(_TAG.sub(" ", text or ""))).casefold()
    tokens = [token for token in _TOKEN.findall(cleaned) if token not in _STOPWORDS]
    return [_STEMMER.stem(token) for token in tokens]


def _windows(tokens: List[str]) -> List[List[str]]:
    """Cut stems into 8-token windows, stride 4. tokens: one passage. Returns a list of windows."""
    if not tokens:
        return []
    if len(tokens) <= _WINDOW:
        return [tokens]
    return [tokens[index : index + _WINDOW] for index in range(0, len(tokens), _STRIDE)]


def _ngrams(tokens: List[str], n: int) -> Counter:
    """Count consecutive n-grams. tokens: fact or one span. n: gram size. Returns a Counter."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _prf(overlap: float, fact_size: float, evidence_size: float) -> tuple:
    """Precision, recall, and F1 from one overlap count. Returns (precision, recall, f1)."""
    precision = overlap / evidence_size if evidence_size else 0.0
    recall = overlap / fact_size if fact_size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _lcs_length(left: List[str], right: List[str]) -> int:
    """Longest common subsequence. left, right: token lists. Returns the length."""
    if not left or not right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for token in right:
        current = [0]
        for index, fact_token in enumerate(left, start=1):
            if token == fact_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


# The checker uses rouge1_recall. The other names exist so the same evidence
# can be scored with the rest of the ROUGE family.
_ROUGE_PART = {
    "rouge1_precision": ("n", 1, 0),
    "rouge1_recall": ("n", 1, 1),
    "rouge1_f1": ("n", 1, 2),
    "rouge2_precision": ("n", 2, 0),
    "rouge2_recall": ("n", 2, 1),
    "rouge2_f1": ("n", 2, 2),
    "rougeL_precision": ("l", 0, 0),
    "rougeL_recall": ("l", 0, 1),
    "rougeL_f1": ("l", 0, 2),
}


def _prepare(fact: str, passages: List[dict], span: str) -> tuple:
    """Tokenise the fact and evidence with title stems removed. span: 'window' or 'page'. Returns (fact_tokens, units)."""
    fact_tokens = _tokenize(fact)
    title_for_fact: List[str] = []
    units: List[List[str]] = []
    for passage in passages:
        title = _tokenize(passage.get("title") or "")
        if title and not title_for_fact:
            title_for_fact = title
        page = [token for token in _tokenize(passage.get("text") or "") if token not in set(title)]
        if span == "page":
            units.append(page)
        else:
            units.extend(_windows(page))
    fact_tokens = [token for token in fact_tokens if token not in set(title_for_fact)]
    return fact_tokens, units


def _unit_prf(fact_tokens: List[str], unit: List[str], family: str, n: int) -> tuple:
    """Score one fact against one span. family: 'n' or 'l'. Returns (precision, recall, f1)."""
    if family == "n":
        fact_counts = _ngrams(fact_tokens, n)
        unit_counts = _ngrams(unit, n)
        overlap = sum(min(count, unit_counts[gram]) for gram, count in fact_counts.items())
        return _prf(overlap, sum(fact_counts.values()), sum(unit_counts.values()))
    return _prf(_lcs_length(fact_tokens, unit), len(fact_tokens), len(unit))


def max_rouge(fact: str, passages: List[dict], variant: str = "rouge1_recall", span: str = "window") -> float:
    """Highest score of one ROUGE variant. span: 'window' or 'page'. Returns a float from 0 to 1."""
    if variant not in _ROUGE_PART:
        raise ValueError("Unknown ROUGE variant: " + variant)
    if span not in ("window", "page"):
        raise ValueError("Unknown span: " + span)
    family, n, part = _ROUGE_PART[variant]
    fact_tokens, units = _prepare(fact, passages, span)
    if not fact_tokens:
        return 0.0
    best = 0.0
    for unit in units:
        if not unit:
            continue
        score = _unit_prf(fact_tokens, unit, family, n)[part]
        if score > best:
            best = score
            if best >= 1.0:
                return 1.0
    return best


def rouge1_recall(fact: str, passages: List[dict]) -> float:
    """Part 1 score: maximum ROUGE-1 recall over 8-token windows. Returns a float."""
    return max_rouge(fact, passages, "rouge1_recall", "window")


class FactExample(object):
    """One fact, its retrieved passages, and the human label."""

    def __init__(self, fact: str, passages: List[dict], label: str):
        """fact: claim. passages: dicts with title and text. label: S, NS, or IR."""
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        """Returns a debug string of the fact, label, and passages."""
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class FactChecker(object):
    """Base checker. Subclasses implement predict."""

    def predict(self, fact: str, passages: List[dict]) -> str:
        """fact: claim. passages: title/text dicts. Returns 'S' or 'NS'."""
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    """Baseline that ignores the inputs and guesses."""

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Returns 'S' or 'NS' at random. fact and passages are unused."""
        import random

        return random.choice(["S", "NS"])


class AlwaysEntailedFactChecker(FactChecker):
    """Baseline that always predicts supported."""

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Returns 'S'. fact and passages are unused."""
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    """Part 1. Supported when ROUGE-1 recall clears the cut."""

    def __init__(self, threshold: float = _ROUGE1_RECALL_THRESHOLD):
        """threshold: minimum ROUGE-1 recall. Default is 3/7."""
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Returns 'S' if rouge1_recall reaches threshold, else 'NS'."""
        return "S" if rouge1_recall(fact, passages) >= self.threshold else "NS"


def _plain(text: str) -> str:
    """Clean one chunk for the model and keep its case. Returns a stripped string."""
    cleaned = unicodedata.normalize("NFKC", html.unescape(_TAG.sub(" ", text or "")))
    return re.sub(r"\s+", " ", cleaned).strip()


def _with_title(title: str, text: str) -> str:
    """Prefix title when it is missing, so a pronoun has a referent. Returns the premise string."""
    if title and title.casefold() not in text.casefold():
        return f"{title}. {text}"
    return text


def iter_evidence(fact: str, passages: List[dict]) -> List[dict]:
    """Build sentence, pair, and passage premises. Returns dicts with kind, overlap, body, and premise."""
    rows = []
    seen = set()
    for passage in passages:
        title = passage.get("title") or ""
        chunks = _SENTENCE.findall(passage.get("text") or "")
        if not chunks:
            chunks = [passage.get("text") or ""]
        sentences = []
        for chunk in chunks:
            sentence = _plain(chunk)
            key = sentence.casefold()
            if len(sentence) < 8 or key in _BOILERPLATE or key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
            rows.append(
                {
                    "kind": "sentence",
                    "overlap": max_rouge(fact, [{"title": title, "text": sentence}], "rouge1_recall", "page"),
                    "body": sentence,
                    "premise": _with_title(title, sentence),
                }
            )
        for index in range(len(sentences) - 1):
            paired = sentences[index] + " " + sentences[index + 1]
            rows.append(
                {
                    "kind": "pair",
                    "overlap": max_rouge(fact, [{"title": title, "text": paired}], "rouge1_recall", "page"),
                    "body": paired,
                    "premise": _with_title(title, paired),
                }
            )
        page = _plain(passage.get("text") or "")
        if len(page) >= 8:
            rows.append(
                {
                    "kind": "passage",
                    "overlap": max_rouge(fact, [{"title": title, "text": page}], "rouge1_recall", "page"),
                    "body": page[:1500],
                    "premise": _with_title(title, page[:1500]),
                }
            )
    return rows


def decide_support(rows: Sequence[dict], kinds: Sequence[str], prune: float, limit: int, decision: str, threshold: float) -> str:
    """Drop low-overlap premises, then apply max_prob or discrete. Returns 'S' or 'NS'."""
    chosen = [row for row in rows if row["kind"] in kinds and row["overlap"] >= prune]
    chosen.sort(key=lambda row: row["overlap"], reverse=True)
    chosen = chosen[:limit]
    if not chosen:
        return "NS"
    if decision == "discrete":
        return "S" if any(row["nli"] == "entailment" for row in chosen) else "NS"
    best = max(float(row["prob"]) for row in chosen)
    return "S" if best >= threshold else "NS"


class EntailmentModel(object):
    """Runs the shared entailment model and reads the entailment probability."""

    def __init__(self, model, tokenizer, cuda=False):
        """model, tokenizer: loaded by the driver. cuda: score on GPU when True."""
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        if self.model is not None:
            self.model.eval()
            id2label = self.model.config.id2label
            self.labels = [id2label[index] if index in id2label else id2label[str(index)] for index in range(len(id2label))]
            self.entailment_index = self.labels.index("entailment")

    def check_entailment(self, premise: str, hypothesis: str):
        """Score one premise against the fact. Returns (entailment probability, label)."""
        prob, label = self.check_entailment_batch([premise], hypothesis)[0]
        return prob, label

    def check_entailment_batch(self, premises: Sequence[str], hypothesis: str):
        """Score many premises against one fact. Returns (probability, label) pairs in order."""
        import torch

        results = []
        for start in range(0, len(premises), _NLI_BATCH):
            batch = list(premises[start : start + _NLI_BATCH])
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    [hypothesis] * len(batch),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                if self.cuda:
                    inputs = {key: value.to("cuda") for key, value in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).detach().cpu()
            for row in probabilities:
                predicted = int(torch.argmax(row).item())
                results.append((float(row[self.entailment_index].item()), self.labels[predicted]))
            del inputs, outputs, logits, probabilities
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
        return results


class EntailmentFactChecker(FactChecker):
    """Part 2. Gate premises by overlap, then take the maximum entailment probability."""

    def __init__(self, ent_model, threshold: float = _ENTAILMENT_THRESHOLD):
        """ent_model: an EntailmentModel. threshold: minimum entailment probability. Default 0.25."""
        self.ent_model = ent_model
        self.threshold = threshold

    def score_evidence(self, fact: str, passages: List[dict]) -> List[dict]:
        """Score every premise, including those below the gate. Returns rows with prob and nli."""
        rows = iter_evidence(fact, passages)
        if not rows:
            return []
        scored = self.ent_model.check_entailment_batch([row["premise"] for row in rows], fact)
        for row, (prob, label) in zip(rows, scored):
            row["prob"] = prob
            row["nli"] = label
        return rows

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Gate by overlap, score what remains, and return 'S' or 'NS'."""
        rows = [
            row
            for row in iter_evidence(fact, passages)
            if row["kind"] in _EVIDENCE_KINDS and row["overlap"] >= _PRUNE_OVERLAP
        ]
        rows.sort(key=lambda row: row["overlap"], reverse=True)
        rows = rows[:_MAX_PREMISES]
        if not rows:
            return "NS"
        scored = self.ent_model.check_entailment_batch([row["premise"] for row in rows], fact)
        for row, (prob, label) in zip(rows, scored):
            row["prob"] = prob
            row["nli"] = label
        return decide_support(rows, _EVIDENCE_KINDS, _PRUNE_OVERLAP, _MAX_PREMISES, _DECISION, self.threshold)


class DependencyRecallThresholdFactChecker(FactChecker):
    """Unused. Parsing mode is not part of this submission."""

    def predict(self, fact: str, passages: List[dict]) -> str:
        """Not implemented. Raises NotImplementedError."""
        raise NotImplementedError("Dependency overlap is not part of this submission")


# The block below runs only as `python factcheck.py`. The coursework driver imports
# the classes above and does not enter it. Every number printed here is computed
# in this file from the label and passage files.


def _report_paths():
    """Locate the coursework jsonl files next to this file. Returns (labels, passages)."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    candidates = (
        (here / "data" / "dev_labeled_ChatGPT.jsonl", here / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"),
        (here.parent / "dev_labeled_ChatGPT.jsonl", here.parent / "passages_bm25_ChatGPT_humfacts.jsonl"),
    )
    for labels, passages in candidates:
        if labels.is_file() and passages.is_file():
            return labels, passages
    raise FileNotFoundError("Put dev_labeled_ChatGPT.jsonl and passages_bm25_ChatGPT_humfacts.jsonl in code/data.")


def _load_report_examples():
    """Read the 221 labelled facts and their passages. Returns a list of dicts."""
    import json

    labels_path, passages_path = _report_paths()
    passages = {}
    with passages_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            passages[row["sent"]] = row["passages"]
    examples = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("annotations") is None:
                continue
            biography = row.get("topic") or ""
            for sentence in row["annotations"]:
                facts = sentence.get("human-atomic-facts")
                if not facts:
                    continue
                for fact in facts:
                    raw = fact["label"]
                    examples.append(
                        {
                            "fact": fact["text"],
                            "passages": passages[fact["text"]],
                            "raw_label": raw,
                            "gold": "NS" if raw == "IR" else raw,
                            "supported": raw == "S",
                            "biography": biography,
                        }
                    )
    return examples


def _switch_tokens(text: str, stop: bool, stem: bool) -> List[str]:
    """Tokenise with the stop list and stemmer switched independently. Returns stems or words."""
    cleaned = unicodedata.normalize("NFKC", html.unescape(_TAG.sub(" ", text or ""))).casefold()
    found = [token for token in _TOKEN.findall(cleaned) if token not in _STOPWORDS or not stop]
    if stem:
        found = [_STEMMER.stem(token) for token in found]
    return found


def _switch_windows(tokens: List[str]) -> List[List[str]]:
    """Same windows as the checker. tokens: one passage. Returns windows."""
    if not tokens:
        return []
    if len(tokens) <= _WINDOW:
        return [tokens]
    return [tokens[index : index + _WINDOW] for index in range(0, len(tokens), _STRIDE)]


def _rouge1_switch(fact: str, passages: List[dict], stop: bool, stem: bool, title: bool, window: bool) -> float:
    """ROUGE-1 recall with one Part 1 choice turned off. Returns a float from 0 to 1."""
    fact_tokens = _switch_tokens(fact, stop, stem)
    title_for_fact: List[str] = []
    units: List[List[str]] = []
    for passage in passages:
        title_tokens = _switch_tokens(passage.get("title") or "", stop, stem)
        if title_tokens and not title_for_fact:
            title_for_fact = title_tokens
        page = _switch_tokens(passage.get("text") or "", stop, stem)
        if title:
            banned = set(title_tokens)
            page = [token for token in page if token not in banned]
        units.extend(_switch_windows(page) if window else [page])
    if title:
        fact_tokens = [token for token in fact_tokens if token not in set(title_for_fact)]
    if not fact_tokens:
        return 0.0
    fact_counts = _ngrams(fact_tokens, 1)
    fact_size = sum(fact_counts.values())
    best = 0.0
    for unit in units:
        if not unit:
            continue
        unit_counts = _ngrams(unit, 1)
        overlap = sum(min(count, unit_counts[gram]) for gram, count in fact_counts.items())
        recall = overlap / fact_size if fact_size else 0.0
        if recall > best:
            best = recall
            if best >= 1.0:
                return 1.0
    return best


def _counts_at(scores, gold_supported, threshold: float) -> dict:
    """Confusion at one cut. gold_supported: True when the label is S. Returns counts and rates."""
    tp = fp = tn = fn = 0
    for score, supported in zip(scores, gold_supported):
        predicted = score >= threshold
        if predicted and supported:
            tp += 1
        elif predicted:
            fp += 1
        elif supported:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "correct": tp + tn,
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _best_cut(scores, gold_supported) -> dict:
    """Highest cut that maximises accuracy. Returns the confusion at that cut."""
    candidates = sorted(set(scores))
    above = (max(candidates) + 1.0) if candidates else 1.0
    chosen = None
    best_correct = -1
    for threshold in list(candidates) + [above]:
        summary = _counts_at(scores, gold_supported, threshold)
        if summary["correct"] > best_correct or (summary["correct"] == best_correct and threshold > chosen["threshold"]):
            best_correct = summary["correct"]
            chosen = summary
    return chosen


def _assign_folds(biographies) -> dict:
    """Five biography folds, names shuffled with seed 3604. Returns name to fold."""
    import random

    order = sorted(set(biographies))
    random.Random(3604).shuffle(order)
    return {name: index % 5 for index, name in enumerate(order)}


def _held_out(scores, gold_supported, biographies, folds) -> int:
    """Refit the cut on the other biographies and count test facts correct. Returns a count."""
    correct = 0
    for fold in range(5):
        train_s, train_g, test_s, test_g = [], [], [], []
        for score, supported, name in zip(scores, gold_supported, biographies):
            if folds[name] == fold:
                test_s.append(score)
                test_g.append(supported)
            else:
                train_s.append(score)
                train_g.append(supported)
        threshold = _best_cut(train_s, train_g)["threshold"]
        correct += sum((score >= threshold) == supported for score, supported in zip(test_s, test_g))
    return correct


def _ratio(correct: int, total: int = 221) -> str:
    """Format a count as in the report. Returns '183/221 (0.828)'."""
    return f"{correct}/{total} ({correct / total:.3f})"


def _mark(on: bool) -> str:
    """A check when the switch is on. Returns 'Y' or a blank."""
    return "Y" if on else " "


def _print_part1(examples) -> None:
    """Print the Part 1 tables: choices, other overlap scores, and the submitted confusion."""
    gold = [example["supported"] for example in examples]
    biographies = [example["biography"] for example in examples]
    folds = _assign_folds(biographies)
    print("Part 1. ROUGE-1 recall")
    print("Each row turns one choice off and retunes the cut. Y means the choice is on.")
    print(f"{'Stop':4} {'Stem':4} {'Title':5} {'Windows':7} {'Dev accuracy':16} {'Held-out':16} {'P':7} {'R':7} {'F1':7}")
    switches = (
        ("submitted", True, True, True, True),
        ("no stop list", False, True, True, True),
        ("no stemming", True, False, True, True),
        ("no title removal", True, True, False, True),
        ("no windows", True, True, True, False),
    )
    for _name, stop, stem, title, window in switches:
        if stop and stem and title and window:
            scores = [rouge1_recall(example["fact"], example["passages"]) for example in examples]
        else:
            scores = [_rouge1_switch(example["fact"], example["passages"], stop, stem, title, window) for example in examples]
        summary = _best_cut(scores, gold)
        held = _held_out(scores, gold, biographies, folds)
        print(
            f"{_mark(stop):4} {_mark(stem):4} {_mark(title):5} {_mark(window):7} "
            f"{_ratio(summary['correct']):16} {_ratio(held):16} "
            f"{summary['precision']:.3f}   {summary['recall']:.3f}   {summary['f1']:.3f}"
        )
    print()
    print("Other overlap scores, same tokens, windows of 8. The cut is retuned on each row.")
    print(f"{'Score':18} {'Full set':16} {'Held-out':16}")
    for variant, label in (
        ("rouge1_precision", "ROUGE-1 precision"),
        ("rouge2_recall", "ROUGE-2 recall"),
        ("rougeL_recall", "ROUGE-L recall"),
        ("rouge1_recall", "ROUGE-1 recall"),
    ):
        scores = [max_rouge(example["fact"], example["passages"], variant, "window") for example in examples]
        summary = _best_cut(scores, gold)
        held = _held_out(scores, gold, biographies, folds)
        print(f"{label:18} {_ratio(summary['correct']):16} {_ratio(held):16}")
    print()
    scores = [rouge1_recall(example["fact"], example["passages"]) for example in examples]
    fixed = _counts_at(scores, gold, _ROUGE1_RECALL_THRESHOLD)
    held = _held_out(scores, gold, biographies, folds)
    print(f"Cut { _ROUGE1_RECALL_THRESHOLD } = 3/7")
    print("The highest score is the maximum of R(F,W) over the windows.")
    print(f"Correct                  {_ratio(fixed['correct'])}")
    print(f"Supported facts recovered {fixed['tp']} of {fixed['tp'] + fixed['fn']}")
    print(f"Supported facts missed    {fixed['fn']}")
    print(f"Negative facts labelled supported {fixed['fp']} of {fixed['fp'] + fixed['tn']}")
    print(f"Precision                 {fixed['tp']}/{fixed['tp'] + fixed['fp']} ({fixed['precision']:.3f})")
    print(f"Recall                    {fixed['tp']}/{fixed['tp'] + fixed['fn']} ({fixed['recall']:.3f})")
    print(f"F1                        {fixed['f1']:.3f}")
    print(f"Held-out                  {_ratio(held)}")
    print("Target                    0.75")
    print()


def _premise_inventory(examples) -> dict:
    """Count dropped lines and the premises built before the gate. Returns the inventory."""
    short_dropped = boilerplate_dropped = duplicate_dropped = 0
    sentences_kept = pairs_built = passages_built = prefixed = 0
    for example in examples:
        seen = set()
        for passage in example["passages"]:
            chunks = _SENTENCE.findall(passage.get("text") or "")
            if not chunks:
                chunks = [passage.get("text") or ""]
            kept_here = 0
            for chunk in chunks:
                sentence = _plain(chunk)
                key = sentence.casefold()
                if len(sentence) < 8:
                    short_dropped += 1
                    continue
                if key in _BOILERPLATE:
                    boilerplate_dropped += 1
                    continue
                if key in seen:
                    duplicate_dropped += 1
                    continue
                seen.add(key)
                sentences_kept += 1
                kept_here += 1
            if kept_here >= 2:
                pairs_built += kept_here - 1
            page = _plain(passage.get("text") or "")
            if len(page) >= 8:
                passages_built += 1
        for row in example["rows"]:
            if row["premise"] != row["body"]:
                prefixed += 1
    return {
        "short": short_dropped,
        "boilerplate": boilerplate_dropped,
        "repeated": duplicate_dropped,
        "sentences": sentences_kept,
        "pairs": pairs_built,
        "passages": passages_built,
        "before_gate": sentences_kept + pairs_built + passages_built,
        "prefixes": prefixed,
    }


def _select_rows(rows, kinds, prune: float, limit: int):
    """Premises of the requested kinds at or above the gate, highest overlap first. Returns a list."""
    chosen = [row for row in rows if row["kind"] in kinds and row["overlap"] >= prune]
    chosen.sort(key=lambda row: row["overlap"], reverse=True)
    return chosen[:limit]


def _fact_score(chosen, prob_key: str) -> float:
    """Maximum entailment probability, or 0 when no premise remains. Returns a float."""
    if not chosen:
        return 0.0
    return max(float(row[prob_key]) for row in chosen)


def _predict_from(chosen, decision: str, threshold: float, prob_key: str, nli_key: str) -> str:
    """Apply one decision rule. Returns 'S' or 'NS'."""
    if not chosen:
        return "NS"
    if decision == "discrete":
        return "S" if any(row[nli_key] == "entailment" for row in chosen) else "NS"
    if decision == "best_overlap_label":
        best = max(chosen, key=lambda row: row["overlap"])
        return "S" if best[nli_key] == "entailment" else "NS"
    return "S" if _fact_score(chosen, prob_key) >= threshold else "NS"


def _rule_counts(examples, kinds, prune: float, limit: int, decision: str, threshold: float, prob_key: str, nli_key: str) -> dict:
    """Confusion and the number of premises sent to the model. Returns counts."""
    scores = []
    gold = []
    pair_count = 0
    empty_supported = 0
    empty_negative = 0
    tp = fp = tn = fn = 0
    for example in examples:
        chosen = _select_rows(example["rows"], kinds, prune, limit)
        pair_count += len(chosen)
        supported = example["supported"]
        if not chosen:
            if supported:
                empty_supported += 1
            else:
                empty_negative += 1
        predicted = _predict_from(chosen, decision, threshold, prob_key, nli_key)
        scores.append(_fact_score(chosen, prob_key))
        gold.append(supported)
        guessed = predicted == "S"
        if guessed and supported:
            tp += 1
        elif guessed:
            fp += 1
        elif supported:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "correct": tp + tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pairs": pair_count,
        "empty_supported": empty_supported,
        "empty_negative": empty_negative,
        "scores": scores,
        "gold": gold,
    }


def _pairwise(scores, gold) -> float:
    """Share of supported/not-supported pairs in which the supported score is higher. Returns a float."""
    positives = [score for score, supported in zip(scores, gold) if supported]
    negatives = [score for score, supported in zip(scores, gold) if not supported]
    hits = ties = 0
    for left in positives:
        for right in negatives:
            if left > right:
                hits += 1
            elif left == right:
                ties += 1
    return (hits + 0.5 * ties) / (len(positives) * len(negatives))


def _attach_model_scores(examples) -> None:
    """Score every premise, then the unprefixed copy when the title was added."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    cuda = torch.cuda.is_available()
    print(f"Loading {model_name}  cuda {cuda}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if cuda:
        model.to("cuda")
    ent_model = EntailmentModel(model, tokenizer, cuda)
    total = len(examples)
    for index, example in enumerate(examples, start=1):
        rows = example["rows"]
        if rows:
            scored = ent_model.check_entailment_batch([row["premise"] for row in rows], example["fact"])
            for row, (prob, label) in zip(rows, scored):
                row["prob"] = prob
                row["nli"] = label
            pending = [row for row in rows if row["body"] != row["premise"]]
            for row in rows:
                if row["body"] == row["premise"]:
                    row["bare_prob"] = row["prob"]
                    row["bare_nli"] = row["nli"]
            if pending:
                bare = ent_model.check_entailment_batch([row["body"] for row in pending], example["fact"])
                for row, (prob, label) in zip(pending, bare):
                    row["bare_prob"] = prob
                    row["bare_nli"] = label
        if index % 20 == 0 or index == total:
            print(f"  scored {index}/{total}", flush=True)


def _print_part2(examples) -> None:
    """Print the Part 2 tables: premises, the gate, the score, the rules, and the premise kinds."""
    for example in examples:
        example["rows"] = iter_evidence(example["fact"], example["passages"])
    inventory = _premise_inventory(examples)
    _attach_model_scores(examples)
    gold_flags = [example["supported"] for example in examples]
    biographies = [example["biography"] for example in examples]
    folds = _assign_folds(biographies)
    kinds_all = ("sentence", "pair", "passage")
    print()
    print("Part 2. Sentence, pair, and passage entailment")
    print("Premise counts, before the gate")
    print(f"Short sentences dropped     {inventory['short']}")
    print(f"Boilerplate lines dropped   {inventory['boilerplate']}")
    print(f"Repeated sentences dropped  {inventory['repeated']}")
    print(f"Sentences                   {inventory['sentences']}")
    print(f"Adjacent pairs              {inventory['pairs']}")
    print(f"Passage premises            {inventory['passages']}")
    print(f"Premises before the gate    {inventory['before_gate']}")
    print(f"Title prefixes              {inventory['prefixes']}")
    print()

    def show(name: str, kinds, prune: float, limit: int, decision: str = "max_prob", threshold: float = _ENTAILMENT_THRESHOLD, prob_key: str = "prob", nli_key: str = "nli"):
        summary = _rule_counts(examples, kinds, prune, limit, decision, threshold, prob_key, nli_key)
        held = _held_out(summary["scores"], gold_flags, biographies, folds) if decision == "max_prob" else None
        return summary, held

    print("What each premise choice scores, at probability cut 0.25 and overlap gate 0.30")
    for name, kinds in (
        ("Sentences and passage", ("sentence", "passage")),
        ("Passage alone", ("passage",)),
        ("Sentences alone", ("sentence",)),
        ("Without the passage", ("sentence", "pair")),
        ("Without pairs", ("sentence", "passage")),
        ("Pairs alone", ("pair",)),
        ("Without sentences", ("pair", "passage")),
    ):
        summary, _held = show(name, kinds, _PRUNE_OVERLAP, _MAX_PREMISES)
        print(f"{name:24} {_ratio(summary['correct'])}  pairs {summary['pairs']}")
    bare, _held = show("Without the title prefix", kinds_all, _PRUNE_OVERLAP, _MAX_PREMISES, prob_key="bare_prob", nli_key="bare_nli")
    print(f"{'Without the title prefix':24} {_ratio(bare['correct'])}")
    print(f"  Precision {bare['precision']:.3f}   Recall {bare['recall']:.3f}   Misses {bare['fn']}   False supports {bare['fp']}")
    cap8, _held = show("Cap of 8", kinds_all, _PRUNE_OVERLAP, 8)
    print(f"{'Cap of 8':24} {_ratio(cap8['correct'])}  on {cap8['pairs']} premises")
    print()
    print("Overlap gate. Maximum probability, cut 0.25.")
    print(f"{'Setting':10} {'Cut':6} {'Accuracy':16} {'P':7} {'R':7} {'F1':7} {'Pairs':6}")
    for name, prune in (("No skip", 0.0), ("Overlap", 0.15), ("Overlap", 0.30)):
        summary, _held = show(name, kinds_all, prune, _MAX_PREMISES)
        print(
            f"{name:10} {prune:<6.3f} {_ratio(summary['correct']):16} "
            f"{summary['precision']:.3f}   {summary['recall']:.3f}   {summary['f1']:.3f}   {summary['pairs']}"
        )
    official, _held_unused = show("submitted", kinds_all, _PRUNE_OVERLAP, _MAX_PREMISES)
    print(
        f"At overlap 0.30, {official['empty_supported']} supported facts and "
        f"{official['empty_negative']} negative facts have no premise left."
    )
    print()
    held = _held_out(official["scores"], gold_flags, biographies, folds)
    print(f"Score cut {_ENTAILMENT_THRESHOLD}")
    print(f"Accuracy                  {_ratio(official['correct'])}")
    print(f"True supports             {official['tp']}/{official['tp'] + official['fn']}")
    print(f"Misses                    {official['fn']}")
    print(f"False supports            {official['fp']}/{official['fp'] + official['tn']}")
    print(f"Precision                 {official['tp']}/{official['tp'] + official['fp']} ({official['precision']:.3f})")
    print(f"Recall                    {official['tp']}/{official['tp'] + official['fn']} ({official['recall']:.3f})")
    print(f"F1                        {official['f1']:.3f}")
    print(f"Pairwise probability      {_pairwise(official['scores'], gold_flags):.3f}")
    print(f"Held-out                  {_ratio(held)}")
    print("Target                    0.83")
    print()
    print("Decision rules. Same premises, gate 0.30, cap 48.")
    print(f"{'Rule':24} {'Cut':6} {'Accuracy':16} {'Held-out':16} {'P':7} {'R':7} {'F1':7}")
    for name, decision, threshold in (
        ("Maximum probability", "max_prob", _ENTAILMENT_THRESHOLD),
        ("Any entailment label", "discrete", _ENTAILMENT_THRESHOLD),
        ("Highest-overlap label", "best_overlap_label", _ENTAILMENT_THRESHOLD),
    ):
        summary, rule_held = show(name, kinds_all, _PRUNE_OVERLAP, _MAX_PREMISES, decision, threshold)
        held_text = _ratio(rule_held) if rule_held is not None else ""
        cut_text = f"{threshold:.3f}" if decision == "max_prob" else ""
        print(
            f"{name:24} {cut_text:6} {_ratio(summary['correct']):16} {held_text:16} "
            f"{summary['precision']:.3f}   {summary['recall']:.3f}   {summary['f1']:.3f}"
        )
    print()
    print("Premise kinds. Maximum probability, cut 0.25, overlap gate 0.30.")
    print(f"{'Sentence':8} {'Pair':4} {'Passage':7} {'Cap':4} {'Accuracy':16} {'P':7} {'R':7} {'F1':7} {'Pairs':6}")
    for sentence, pair, passage, limit in (
        (True, True, True, 48),
        (False, True, True, 48),
        (True, False, True, 48),
        (True, True, False, 48),
        (True, True, True, 4),
    ):
        kinds = tuple(kind for kind, on in (("sentence", sentence), ("pair", pair), ("passage", passage)) if on)
        summary, _held = show("kinds", kinds, _PRUNE_OVERLAP, limit)
        print(
            f"{_mark(sentence):8} {_mark(pair):4} {_mark(passage):7} {limit:<4} {_ratio(summary['correct']):16} "
            f"{summary['precision']:.3f}   {summary['recall']:.3f}   {summary['f1']:.3f}   {summary['pairs']}"
        )
    print()
    return official


def _print_part3(examples, official) -> None:
    """Print the entailment errors the report reads: 8 false supports and the first 10 misses."""
    print("Part 3. Errors of the entailment checker")
    print(f"False supports {official['fp']}    Misses {official['fn']}")
    print("The sample is every false support and the first 10 misses in labelled-file order.")
    false_supports = []
    misses = []
    for example in examples:
        chosen = _select_rows(example["rows"], ("sentence", "pair", "passage"), _PRUNE_OVERLAP, _MAX_PREMISES)
        predicted = _predict_from(chosen, "max_prob", _ENTAILMENT_THRESHOLD, "prob", "nli")
        if predicted == example["gold"]:
            continue
        best = max(chosen, key=lambda row: float(row["prob"])) if chosen else None
        record = {
            "fact": example["fact"],
            "gold": example["gold"],
            "pred": predicted,
            "raw": example["raw_label"],
            "score": 0.0 if best is None else float(best["prob"]),
            "overlap": 0.0 if best is None else float(best["overlap"]),
            "premise": "" if best is None else best["premise"],
            "direction": "false support" if predicted == "S" else "miss",
        }
        if predicted == "S":
            false_supports.append(record)
        else:
            misses.append(record)
    sample = false_supports + misses[:10]
    print(f"{'Fact':70} {'Gold':4} {'Pred':4} {'Score':8} {'Overlap':8}")
    for record in sample:
        fact = record["fact"] if len(record["fact"]) <= 68 else record["fact"][:65] + "..."
        print(f"{fact:70} {record['gold']:4} {record['pred']:4} {record['score']:8.4f} {record['overlap']:8.4f}")
    print()
    print(f"False supports in the sample that were labelled irrelevant: {sum(row['raw'] == 'IR' for row in false_supports)}")
    named = (
        "Trivisonno worked at WKNR",
        "The Fischer Research Laboratory became one of the world's leading manufacturers of metal detectors",
        "Don Featherstone was an artist",
        "Just Cause is a film",
        "Shahnaz married Ardeshir Zahedi",
    )
    recorded = false_supports + misses
    print("Named facts")
    for fact in named:
        record = next((row for row in recorded if row["fact"].startswith(fact)), None)
        if record is None:
            print(f"  {fact}: not among the errors")
            continue
        print(f"  {fact}")
        print(f"    {record['direction']}, gold {record['gold']}, predicted {record['pred']}, score {record['score']:.4f}, overlap {record['overlap']:.4f}")


def show_report(parts=("all",)) -> None:
    """Print the report results. parts: '1', '2', '3', or 'all'. Part 3 needs the Part 2 scores."""
    selected = {"1", "2", "3"} if "all" in parts else set(parts)
    examples = _load_report_examples()
    print(f"Loaded {len(examples)} facts from the coursework files. This file computed the results below.")
    print()
    if "1" in selected:
        _print_part1(examples)
    if "2" in selected or "3" in selected:
        official = _print_part2(examples)
        if "3" in selected:
            _print_part3(examples, official)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print the report results from this file.")
    parser.add_argument("--part", nargs="+", choices=("1", "2", "3", "all"), default=["1", "2"])
    show_report(parser.parse_args().part)
