# Part 1: Word Overlap with TF-IDF

This note explains the word-overlap fact checker used in Task A: what TF-IDF is, how it is implemented here, and why the final classifier uses **IDF-weighted recall** instead of plain TF-IDF cosine similarity.

## 1. The task

Each example is an atomic **fact** (e.g. *"Marianne McAndrew was born in Cleveland, Ohio."*) paired with Wikipedia **passages** retrieved by BM25. The goal is to predict **S** (supported) or **NS** (not supported). `IR` (irrelevant) is mapped to NS, the same way `factchecking_main.py` does.

Dataset: 221 facts (112 S, 93 NS, 16 IR → 109 NS after mapping). Always predicting S would score 112/221 = **50.7%**.

The idea behind word overlap is simple: a supported fact should share many words with some part of its evidence. A fact that isn't supported should share fewer.

## 2. What TF-IDF is

TF-IDF scores how important a term is to one document, relative to a collection of documents.

### Term frequency (TF)
How often term *t* appears in document *d*. We use **log-scaled TF**, so a word repeated 10 times doesn't count 10× more than a word that appears once:

$$\text{tf}(t, d) = 1 + \log(\text{count}(t, d))$$

### Inverse document frequency (IDF)
Terms that appear in almost every document (like the subject's name, which is in every passage about them) tell us little. Rare terms (like "Cleveland" or "1942") tell us a lot. We use **smoothed IDF**:

$$\text{idf}(t) = \log\frac{N + 1}{\text{df}(t) + 1} + 1$$

where *N* is the number of documents and df(*t*) is the number of documents containing *t*. The `+1` terms avoid division by zero, and the trailing `+1` keeps every weight positive.

### TF-IDF weight and vector
$$w(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

Each document becomes a sparse vector of these weights, which is then **L2-normalised** (scaled to length 1).

### Cosine similarity
For normalised vectors, cosine similarity is just the dot product:

$$\cos(q, d) = \sum_t q_t \, d_t \in [0, 1]$$

1 means the term distributions are identical. 0 means the two documents share no terms.

## 3. How it is applied to fact checking

For each fact:

1. **Build evidence units.** Each passage is either kept whole (*passage level*) or split into sentences on `.`, `!` or `?` followed by whitespace (*sentence level*). `<s>…</s>` tags are stripped first.
2. **Build a small local corpus:** `[fact] + evidence_units`. IDF is computed over this corpus, so a term is "rare" relative to that person's own Wikipedia text.
3. **Score** the fact against every evidence unit and take the **max**. The fact only needs to be supported by one place in the evidence.
4. **Threshold:** predict S if `score ≥ θ`, otherwise NS.

Everything is implemented with plain Python dictionaries (`Counter`, `log`, `sqrt`), with no scikit-learn, so every step of the maths is visible.

## 4. Preprocessing choices

| Step | Choice | Reason |
|---|---|---|
| Lowercasing | Yes | "Ohio" and "ohio" should match. |
| Markup removal | Strip `<…>` tags | Passages contain `<s>` sentence tags. |
| Punctuation | Replaced with spaces; keep `a-z0-9` | Keeps years and dates ("1942") as tokens. |
| Stopwords | spaCy English list **minus negations** (`no`, `not`, `nor`, `never`, `none`, `n't`) | Removes function words like "also", "has" and "which" that inflate overlap. Negations are kept because removing them would make a fact and evidence that contradicts it look more alike. |
| Stemming | Porter stemmer | "acting", "actor" and "acted" collapse towards the same stem, improving recall. |
| N-grams | Unigrams (bigrams tested) | Bigrams are sparse on short facts and lowered accuracy (see §6). |

spaCy's stoplist is used instead of NLTK's because it needs no `nltk.download()`, so `factcheck.py` runs on a fresh machine.

## 5. From cosine to IDF-weighted recall

Cosine similarity is **symmetric**: it penalises the evidence for having lots of words the fact doesn't contain. Facts are short, but evidence sentences and passages are long, so a passage that fully supports a fact can still get a low cosine score just because it says other things too.

The question we actually care about is asymmetric: *how much of the fact is covered by the evidence?* That's **recall**. Weighting each term by IDF means the rare, informative words (names of places, dates, titles) count most, and the subject's name, which appears everywhere, counts least:

$$\text{IDF-recall}(f, u) = \frac{\sum_{t \in F \cap U} \text{idf}(t)}{\sum_{t \in F} \text{idf}(t)}$$

where *F* is the fact's set of terms and *U* is the evidence unit's set of terms. The final score is the max over all evidence units.

This keeps TF-IDF's key ingredient (IDF weighting) while matching the shape of the problem. It also fits the class name the framework gives us: `WordRecallThresholdFactChecker`.

## 6. Results

Thresholds were tuned on the dev set with a grid search over [0, 1] in steps of 0.001, maximising accuracy (F1 breaks ties).

| Configuration | Threshold | Accuracy |
|---|---|---|
| TF-IDF cosine, uni+bigram, small stoplist, stem, sentence | 0.172 | 0.7330 |
| TF-IDF cosine, unigram, small stoplist, stem, sentence | 0.257 | 0.7466 |
| TF-IDF cosine, unigram, **spaCy stoplist**, stem, sentence | 0.270 | 0.7602 |
| TF-IDF cosine, uni+bigram, passage level | 0.082 | 0.7059 |
| TF-IDF cosine, no stopwords or stemming | 0.288 | 0.7014 |
| **IDF-recall**, spaCy stoplist, stem, **sentence** | 0.424 | 0.8009 |
| **IDF-recall**, spaCy stoplist, stem, **passage** | **0.605** | **0.8190** |

Observations:

- **Stopword removal and stemming both help.** Removing both drops cosine accuracy to about 0.70.
- **Bigrams hurt.** Facts are 5–10 tokens long, so exact bigram matches are rare and add noise.
- **Passage level hurts cosine but helps recall.** Under cosine, a long passage dilutes the score. Under recall, a longer unit can only cover *more* of the fact, and many facts are split across sentences (e.g. a pronoun "She…" in one sentence and the name in the previous one).
- **IDF-recall beats every cosine variant by about 6 points.**

## 7. Threshold sensitivity

A good method shouldn't depend on hitting one exact threshold.

| Method | Accuracy at θ − 0.04, θ − 0.02, **θ**, θ + 0.02, θ + 0.04 |
|---|---|
| Cosine (spaCy, sentence), θ = 0.270 | 0.697, 0.738, **0.760**, 0.724, 0.719 |
| IDF-recall (spaCy, passage), θ = 0.605 | 0.801, 0.801, **0.819**, 0.810, 0.810 |

Cosine has a **sharp peak**: a small shift costs 2–6 points. IDF-recall sits on a **plateau** of roughly 0.80–0.82, so the chosen threshold is much less likely to be overfit to this dev set. That matters because, as the brief notes, we tune and evaluate on the same set.

## 8. Limitations

- **Bag of words ignores meaning and order.** "Born in Ohio, died in California" overlaps with a fact claiming the reverse.
- **Wrong details still match.** "Born on November 21, 1942" shares "born", "november" and "1942" with evidence saying November 2**2**, 1942, so it scores high but is NS.
- **Paraphrase is missed.** "Actress" vs "appeared in films" share no stems, which causes false negatives.
- **Same set for tuning and evaluation.** The reported accuracy is optimistic; the plateau in §7 reduces, but does not remove, that risk.

These are the gaps Part 2 (textual entailment with DeBERTa-v3) is meant to close. The IDF-recall score is also reused there as a cheap **pruning** filter: facts with very low overlap are labelled NS without running the entailment model.
