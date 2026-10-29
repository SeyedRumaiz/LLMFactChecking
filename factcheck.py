# factcheck.py

import torch
from typing import List
import numpy as np
import json
from pathlib import Path
import spacy
import gc
import re
from collections import Counter
from math import log
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS


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


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda


    def check_entailment(self, premise: str, hypothesis: str):
        """
        Returns the probabilities for one (premise, hylpothesis) pair as a numpy array
        [P(entailment), P(neutral), P(contradiction)].
        """
        return self.check_entailment_batch([premise], [hypothesis])[0]


    def check_entailment_batch(self, premises: List[str], hypotheses: List[str]):
        """
        Batched version: returns an (n, 3) numpy array of [P(entailment), P(neutral), P(contradiction)].
        Batching many pairs per forward pass is much faster on CPU than one pair at a time.
        """
        with torch.no_grad():
            # tokenize the premise and hypothesis pairs
            inputs = self.tokenizer(premises, hypotheses, return_tensors='pt',
                                    truncation=True, max_length=512, padding=True)

            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's predictions
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Labels ["entailment", "neutral", "contradiction"] 
            # softmax turns the logits into probabilities.
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Delete object inputs, outputs and logits since no longer needed after computation
        del inputs, outputs, logits
        gc.collect()

        return probs

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
    """
    IDF-weighted word recall (Part 1 final model).

    score(fact) = max over evidence sentences of
                  sum(idf of fact terms found in the sentence) / sum(idf of all fact terms)

    Improvements over plain IDF-recall:
      1. Subject-name terms (from the Wikipedia page title) are removed from the fact,
         because every retrieved passage contains them; they inflate the score of
         unsupported facts without carrying evidence.
      2. IDF is estimated over the whole retrieval corpus (all unique passages), which is
         far more reliable than a per-fact corpus of 2-5 passages. Falls back to a
         per-fact (local) IDF if the corpus file is not found.
      3. Evidence is scored sentence by sentence.
    Predict S if score >= threshold (tuned on dev: 0.457 -> 87.8%).
    """

    NEGATION_WORDS = {"no", "not", "nor", "never", "none", "n't"}
    PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

    def __init__(self, threshold: float = 0.457, passages_path: str = PASSAGES_PATH):
        self.threshold = threshold
        self.stemmer = PorterStemmer()
        self.stopwords = set(STOP_WORDS) - self.NEGATION_WORDS  # common words to ignore
        self.global_idf = self._build_global_idf(passages_path) # dictionary of word importances

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)    # strip <s> tags
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep letters and digits (years)
        return [self.stemmer.stem(t) for t in text.split() if t not in self.stopwords]


    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r"<[^>]+>", " ", text)

        # splits text at whitespace immediately following a full stop, exclamation mark or question mark
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


    def _build_global_idf(self, path: str):
        """
        Smoothed IDF over every unique passage in the retrieval corpus (no labels used).
        """

        if not Path(path).exists():
            return None
        unique_passages = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for p in json.loads(line)["passages"]:
                    unique_passages[(p["title"], p["text"])] = set(self.tokenize(p["text"]))

        df = Counter()  # counts number of passages containing each term
        for terms in unique_passages.values():
            df.update(terms)
        self.n_docs = len(unique_passages)

        return {t: log((self.n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        


    def idf(self, term: str, local_idf=None) -> float:
        if self.global_idf is not None:
            # Unseen term = appears in 0 documents.
            return self.global_idf.get(term, log(self.n_docs + 1) + 1.0)
        return local_idf[term]



    def score(self, fact: str, passages: List[dict]) -> float:
        fact_terms = set(self.tokenize(fact))

        # Remove subject-name terms taken from the page title.
        if passages:
            title = re.sub(r"\(.*?\)", " ", passages[0].get("title", ""))   # drop "(footballer, born 1984)"
            content_terms = fact_terms - set(self.tokenize(title))
            if content_terms:            # keep the name if it's all the fact contains
                fact_terms = content_terms
        if not fact_terms:
            return 0.0

        sentence_term_sets = []
        for p in passages:
            for sent in self.split_into_sentences(p.get("text", "") or ""):
                terms = set(self.tokenize(sent))
                if terms:
                    sentence_term_sets.append(terms)
        if not sentence_term_sets:
            return 0.0

        local_idf = None
        if self.global_idf is None:     # fallback: IDF over [fact] + this fact's sentences
            docs = [fact_terms] + sentence_term_sets
            df = Counter()
            for d in docs:
                df.update(d)
            local_idf = {t: log((len(docs) + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        weights = {t: self.idf(t, local_idf) for t in fact_terms}
        total = sum(weights.values())
        return max(sum(weights[t] for t in fact_terms & s) / total for s in sentence_term_sets)

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if self.score(fact, passages) >= self.threshold else "NS"



class EntailmentFactChecker(FactChecker):
    """
    Textual entailment with DeBERTa-v3 (MNLI + FEVER + ANLI), Part 2 final model.

    1. Pruning: if the Part 1 word-overlap score is below prune_threshold, predict NS
       without running the model (entailment is extremely unlikely with almost no overlap).
    2. Each passage is split into sentences (on <s> tags and . ! ?); fragments < 3 words are dropped.
    3. Premise = previous sentence + current sentence (2-sentence window), so pronoun sentences
       ("He was born in ...") keep the name from the sentence before.
    4. Fact score = max P(entailment) over all premises; predict S if score >= threshold.
       Early exit: stop as soon as one premise reaches the threshold.
    Tuned on dev: prune_threshold = 0.1, threshold = 0.325 -> ~86.4% (5-fold CV ~85.5%).
    """

    def __init__(self, ent_model, threshold: float = 0.325, prune_threshold: float = 0.0, batch_size: int = 16):
        self.ent_model = ent_model
        self.threshold = threshold
        self.prune_threshold = prune_threshold
        self.batch_size = batch_size
        self.overlap_checker = WordRecallThresholdFactChecker()   # Part 1 scorer, reused for pruning

    @staticmethod
    def split_passage(text: str) -> List[str]:
        """Split on the <s>...</s> markers, then on sentence punctuation. Drop tiny fragments."""
        sentences = []
        for chunk in re.split(r"</?s>", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                sent = re.sub(r"\s+", " ", sent).strip()
                if len(sent.split()) >= 3:        # drop noise like "Career."
                    sentences.append(sent)
        return sentences

    def build_premises(self, passages: List[dict]) -> List[str]:
        """2-sentence windows (previous + current sentence) within each passage."""
        premises = []
        for passage in passages:
            sents = self.split_passage(passage.get("text", "") or "")
            for i, sent in enumerate(sents):
                premises.append(sents[i - 1] + " " + sent if i > 0 else sent)
        return premises

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Pruning with word overlap: skip the model when overlap is very low.
        if self.overlap_checker.score(fact, passages) < self.prune_threshold:
            return "NS"

        premises = self.build_premises(passages)
        if not premises:
            return "NS"

        # Run the model in small batches; stop early once any premise entails the fact.
        for start in range(0, len(premises), self.batch_size):
            batch = premises[start:start + self.batch_size]
            probs = self.ent_model.check_entailment_batch(batch, [fact] * len(batch))
            best_entailment = float(probs[:, 0].max())     # column 0 = P(entailment)
            del probs
            if best_entailment >= self.threshold:
                return "S"
        return "NS"


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
