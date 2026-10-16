# factcheck.py

import torch
from typing import List
import numpy as np
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
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


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
    IDF-weighted word recall at passage level.
    score = sum(idf of fact terms found in a passage) / sum(idf of all fact terms),
    max over passages. Predict S if score >= threshold (tuned on dev: 0.605 -> 81.9%).
    """

    NEGATION_WORDS = {"no", "not", "nor", "never", "none", "n't"}

    def __init__(self, threshold: float = 0.605):
        self.threshold = threshold
        self.stemmer = PorterStemmer()
        # Keep negations: removing them would make contradictory evidence look similar.
        self.stopwords = set(STOP_WORDS) - self.NEGATION_WORDS

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)    # strip <s> tags
        text = re.sub(r"[^a-z0-9\s]", " ", text)    # keep letters and digits (years)
        return [self.stemmer.stem(t) for t in text.split() if t not in self.stopwords]

    def score(self, fact: str, passages: List[dict]) -> float:
        fact_terms = set(self.tokenize(fact))
        if not fact_terms:
            return 0.0

        passage_term_sets = [set(self.tokenize(p.get("text", "") or "")) for p in passages]
        passage_term_sets = [s for s in passage_term_sets if s]
        if not passage_term_sets:
            return 0.0

        # Smoothed IDF over [fact] + this fact's passages.
        documents = [fact_terms] + passage_term_sets
        df = Counter()
        for doc in documents:
            df.update(doc)
        n = len(documents)
        idf = {t: log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        total = sum(idf[t] for t in fact_terms)
        return max(sum(idf[t] for t in fact_terms & s) / total for s in passage_term_sets)

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if self.score(fact, passages) >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


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
