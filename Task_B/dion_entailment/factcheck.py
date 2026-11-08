# factcheck.py

import gc
import html
import math
import re
import unicodedata
from typing import List, Sequence, Tuple

import numpy as np
import spacy
import torch
from nltk.stem import PorterStemmer


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


# Function words that do not discriminate support. Negations and numerals are kept.
_STOPWORDS = frozenset(
    "a an the is are was were be been being am to of in on at by for from with as and or but "
    "that this these those it its his her their he she they who which has have had do does did".split()
)
_STEMMER = PorterStemmer()

# Frozen from the Part 1 nested grouped-CV study: Porter stemming, title-token
# stripping, 8-token evidence windows, and unigram recall of the fact.
_WORD_OVERLAP_THRESHOLD = 0.35416666666666663
_WINDOW_SIZE = 8
_WINDOW_STRIDE = 4

# Fact-level overlap pruning at 0.05/0.12 drops eight gold-supported facts that
# have zero lexical overlap, so we do not discard whole facts. Pruning is applied
# to noisy/duplicate sentences and by capping the number of NLI premises.
_PRUNE_OVERLAP_THRESHOLD = 0.0
_MAX_NLI_PREMISES = 48
_NLI_BATCH_SIZE = 4
# Discrete argmax-entailment matched a P(entailment) threshold of 0.5 on this data.
_ENTAILMENT_THRESHOLD = 0.5
_NLI_LABELS = ("entailment", "neutral", "contradiction")


def _normalize(text: str) -> Tuple[str, ...]:
    cleaned = unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).casefold()
    tokens = re.findall(r"[^\W_]+", cleaned)
    return tuple(_STEMMER.stem(tok) for tok in tokens if tok not in _STOPWORDS)


def _content_tokens(text: str, title_tokens: set) -> Tuple[str, ...]:
    return tuple(tok for tok in _normalize(text) if tok not in title_tokens)


def _title_tokens(passages: List[dict]) -> set:
    if not passages:
        return set()
    return set(_normalize(passages[0].get("title") or ""))


def _window_units(tokens: Sequence[str]) -> List[Tuple[str, ...]]:
    if not tokens:
        return []
    if len(tokens) <= _WINDOW_SIZE:
        return [tuple(tokens)]
    return [tuple(tokens[i:i + _WINDOW_SIZE]) for i in range(0, len(tokens), _WINDOW_STRIDE)]


def overlap_score(fact: str, passages: List[dict]) -> float:
    """Unigram recall of the fact against the best 8-token evidence window."""
    title_tokens = _title_tokens(passages)
    fact_tokens = _content_tokens(fact, title_tokens)
    if not fact_tokens:
        return 0.0
    fact_set = set(fact_tokens)
    best = 0.0
    for passage in passages:
        for unit in _window_units(_content_tokens(passage.get("text") or "", title_tokens)):
            if not unit:
                continue
            best = max(best, len(fact_set & set(unit)) / len(fact_set))
            if best >= 1.0:
                return best
    return best


def sentence_units(passages: List[dict]) -> List[str]:
    """Split BM25 Wikipedia chunks on <s> markers and sentence boundaries; drop markup/noise."""
    sentences = []
    seen = set()
    for passage in passages:
        raw = passage.get("text") or ""
        for chunk in re.split(r"</?s>|(?<=[.!?])\s+", raw):
            sentence = unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", " ", chunk)))
            sentence = re.sub(r"\s+", " ", sentence).strip(" \t\n\r-–—")
            if len(sentence) < 8:
                continue
            lowered = sentence.casefold()
            if lowered in {"references.", "see also.", "external links.", "notes."}:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            sentences.append(sentence)
    return sentences


def three_class_to_supported(
    logits: Sequence[float],
    strategy: str = "threshold",
    threshold: float = _ENTAILMENT_THRESHOLD,
) -> str:
    """Map MNLI-style [entailment, neutral, contradiction] logits to S/NS."""
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps) or 1.0
    probs = [value / total for value in exps]
    if strategy == "discrete":
        return "S" if int(np.argmax(probs)) == 0 else "NS"
    return "S" if probs[0] >= threshold else "NS"


def _with_title(title: str, text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    if title and title.casefold() not in text[: max(40, len(title) + 10)].casefold():
        return f"{title}. {text}"
    return text


def nli_premises(passages: List[dict]) -> List[str]:
    """Sentence units, two-sentence windows, and truncated passage chunks, with page titles."""
    title = passages[0].get("title") if passages else ""
    sentences = sentence_units(passages)
    candidates = list(sentences)
    for idx in range(len(sentences) - 1):
        candidates.append(sentences[idx] + " " + sentences[idx + 1])
    for passage in passages[:2]:
        raw = unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", " ", passage.get("text") or "")))
        candidates.append(re.sub(r"\s+", " ", raw).strip()[:1000])

    premises = []
    seen = set()
    for candidate in candidates:
        premise = _with_title(title, candidate)
        key = premise.casefold()
        if len(premise) < 8 or key in seen:
            continue
        seen.add(key)
        premises.append(premise)
        if len(premises) >= _MAX_NLI_PREMISES:
            break
    return premises


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        if self.model is not None:
            self.model.eval()

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            if self.cuda:
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
            predicted_idx = int(torch.argmax(probabilities).item())
            entailment_prob = float(probabilities[0].item())
            predicted_label = _NLI_LABELS[predicted_idx]

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probabilities
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()

        return entailment_prob, predicted_label

    def check_entailment_batch(self, premises: Sequence[str], hypothesis: str):
        """Score several premises against one fact, freeing GPU memory after each mini-batch."""
        results = []
        for start in range(0, len(premises), _NLI_BATCH_SIZE):
            batch = list(premises[start:start + _NLI_BATCH_SIZE])
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
                predicted_idx = int(torch.argmax(row).item())
                results.append((float(row[0].item()), _NLI_LABELS[predicted_idx]))
            del inputs, outputs, logits, probabilities
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
        return results


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = _WORD_OVERLAP_THRESHOLD):
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if overlap_score(fact, passages) >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        premises = nli_premises(passages)
        if not premises:
            return "NS"

        if hasattr(self.ent_model, "check_entailment_batch"):
            scored = self.ent_model.check_entailment_batch(premises, fact)
        else:
            scored = [self.ent_model.check_entailment(premise, fact) for premise in premises]

        best_score = 0.0
        for entailment_prob, predicted_label in scored:
            best_score = max(best_score, float(entailment_prob))
            if predicted_label == "entailment":
                return "S"
        return "S" if best_score >= _ENTAILMENT_THRESHOLD else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
