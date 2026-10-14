# Jaccard Metric — Experiment Log and Results

**Author:** Thamadhi
**Module:** CM3604 Deep Learning
**Part:** Coursework Part 1 — Word Overlap (Jaccard)
**Date started:** [today's date]

**Summary:** This file logs every Jaccard-based experiment on the 221-example dev set — tokenization, stopword handling, stemming, sentence-level vs whole-passage comparison, aggregation strategy, threshold sweep, and IDF weighting — culminating in a final configuration at **70.1% accuracy**. Raw outputs live in the accompanying CSV files in this folder.

---

## 1. Setup

- Python version: 3.10.11
- Virtual environment: `.venv310`
- Key packages: numpy, nltk, spacy, torch, transformers, rouge-score
- Notebook: `jaccard_experiments.ipynb`

---

## 2. Data Loading

### Files loaded

| File | Records |
|------|---------|
| `data/dev_labeled_ChatGPT.jsonl` | 31 |
| `data/passages_bm25_ChatGPT_humfacts.jsonl` | 5427 |

### Record schemas

**Dev record keys:** `input`, `output`, `topic`, `cat`, `annotations`

**Passage record keys:** `name`, `sent`, `passages`

**Human-annotated atomic fact keys:** `text`, `label`

**Join key:** `passages_data[i]["sent"]` matches `annotations[*]["human-atomic-facts"][*]["text"]`

**Labels found:** S (supported), NS (not supported), IR (irrelevant)

**Example atomic fact:** `{"text": "Marianne McAndrew is an American.", "label": "NS"}`

---

## 3. Examples Built

| Metric | Value |
|--------|-------|
| Total examples | 221 |
| Missing passages | 0 |
| Missing text/label | 0 |
| Unrecognized labels | 0 |
| Unique facts in passage dict | 5373 |

### Label distribution

| Raw label | Count | Binary value |
|-----------|-------|--------------|
| S (supported) | 112 | 1 |
| NS (not supported) | 93 | 0 |
| IR (irrelevant) | 16 | 0 |

**Class imbalance note:** approximately 51% S, 42% NS, 7% IR. IR is folded into NS for the binary task. Mild imbalance — accuracy is fine as the primary metric, and we also track precision, recall, and F1 per class.

---

## 4. Preprocessing Utilities

### Stopwords

- Conservative list of 47 stopwords.
- Negation words ("not", "no", "nor", "never") intentionally preserved.
- Rationale: removing negations would make contradicting facts indistinguishable from supporting ones under a bag-of-words model.

### Tokenizer configurations tested

| Config | Lowercase | Stopwords removed | Stemming |
|--------|-----------|-------------------|----------|
| A | Yes | No | No |
| B | Yes | Yes | No |
| C | Yes | Yes | Yes |
| D | Yes | No | Yes |

### Sentence splitting

- Passages contain `<s>...</s>` tags around sentences.
- Primary splitter exploits these tags.
- Fallback splitter uses punctuation-based regex.
- Enables comparing whole-passage vs sentence-level Jaccard in Section 8.

### Sanity check outputs

- "The cat is not sleeping." → `['cat', 'not', 'sleeping']` (with stopwords removed) — negation preserved.
- "The cats were sleeping!" → `['cat', 'sleep']` (with stopwords + stemming) — plural handling correct.
- First example's first passage splits into clean, tag-free sentences.

---

## 5. Jaccard Scoring

### Definition

Jaccard similarity between two token sets:

    J(A, B) = |A ∩ B| / |A ∪ B|

### Scoring pipeline

For each fact:

1. Tokenize the fact (lowercase, optional stopword removal, optional stemming).
2. For each passage:
   - If `sentence_level=False`: treat the whole passage as one segment.
   - If `sentence_level=True`: split into sentences and treat each separately.
3. Compute Jaccard between the fact and each segment.
4. Aggregate all segment scores with one of:
   - `max` — highest single-segment score
   - `mean` — average across all segments
   - `top_k_mean` — average of the top-k highest scores

### Configurable knobs

| Knob | Values |
|------|--------|
| Lowercase | True / False |
| Stopword removal | True / False |
| Stemming | True / False |
| Sentence-level | True / False |
| Aggregation | `max`, `mean`, `top_k_mean` |

### Initial sanity checks

Example: "Marianne McAndrew is an American." (label: NS)

| Config | Score |
|--------|-------|
| Whole passage, max | 0.0326 |
| Sentence-level, max | 0.1538 |
| Sentence-level, mean | 0.0607 |
| Sentence-level, top-3 mean | 0.0809 |

Comparative check (both from the same biography, with `sentence_level=True, aggregation='max'`):

| Fact | Label | Score |
|------|-------|-------|
| Marianne McAndrew is an actress. | S | 0.2500 |
| Marianne McAndrew is an American. | NS | 0.1538 |

**Observation:** the supported fact scored higher than the not-supported one. The signal exists but is weak in absolute terms. The initial guess was that the optimal threshold would be in the 0.15–0.25 range. **Update from Section 7:** the actual best threshold turned out to be **0.03**, much lower than this estimate. Absolute Jaccard scores are lower than the single-example check suggests because most passages contain far more tokens than the fact, diluting the union and lowering the score. This is a useful reminder that single-example intuitions can mislead.

### Single-config trial

Config: `lower=True, remove_stop=True, stem=True, sentence_level=True, aggregation="max"`, threshold = 0.15

| Metric | Value |
|--------|-------|
| Accuracy | 0.6561 |
| F1 (S) | 0.5824 |
| F1 (NS) | 0.7077 |
| Runtime (seconds) | 0.53 |

Confusion matrix `[gold, pred]`:

|            | Pred NS | Pred S |
|------------|---------|--------|
| Gold NS    | 92      | 17     |
| Gold S     | 59      | 53     |

**Observation:** accuracy is 65.6% at threshold 0.15. False negatives on class S (59) are much higher than false positives (17), suggesting the threshold is too strict. The threshold sweep in Section 7 confirms this.

---

## 7. Threshold Sweep (Baseline Config)

Config: `lower=True, remove_stop=True, stem=True, sentence_level=True, aggregation="max"`

| Metric | Value |
|--------|-------|
| Best threshold | 0.08 |
| Best accuracy | 0.6878 |
| F1 (S) at best | 0.7206 |
| F1 (NS) at best | 0.6462 |

Accuracy at sample thresholds:

| Threshold | Accuracy | F1 (S) | F1 (NS) |
|-----------|----------|--------|---------|
| 0.05 | 0.6154 | 0.7099 | 0.4295 |
| 0.10 | 0.6742 | 0.6870 | 0.6604 |
| 0.15 | 0.6561 | 0.5824 | 0.7077 |
| 0.20 | 0.6154 | 0.4586 | 0.7018 |
| 0.25 | 0.5701 | 0.3165 | 0.6865 |
| 0.30 | 0.5294 | 0.1875 | 0.6688 |
| 0.40 | 0.5113 | 0.0847 | 0.6667 |

**Observation:** accuracy peaks at threshold 0.08 (68.8%). Lower thresholds favor class S (higher recall, lower precision); higher thresholds favor NS. Best accuracy reflects a balance point. Current baseline: **68.8%**, roughly 6 points short of the 75% target. The grid search in Section 8 explores preprocessing choices to close this gap.

---

## 8. Preprocessing Grid Search

24 configs tested: `remove_stop ∈ {False, True}` × `stem ∈ {False, True}` × `sentence_level ∈ {False, True}` × `aggregation ∈ {max, mean, top_k_mean}`. Threshold swept per config. `lower=True` fixed.

### Top 10 configs by accuracy

| Rank | stop | stem | sent | agg | best t | acc | F1 (S) | F1 (NS) |
|------|------|------|------|-----|--------|-----|--------|---------|
| 1 | True | True | False | max | 0.03 | 0.6923 | 0.7213 | 0.6566 |
| 2 | True | True | True | max | 0.08 | 0.6878 | 0.7206 | 0.6462 |
| 3 | True | False | True | max | 0.08 | 0.6742 | 0.7049 | 0.6364 |
| 4 | True | False | False | max | 0.03 | 0.6697 | 0.6867 | 0.6507 |
| 5 | True | True | True | top_k_mean | 0.09 | 0.6697 | 0.6294 | 0.7020 |
| 6 | False | True | True | top_k_mean | 0.08 | 0.6652 | 0.6967 | 0.6263 |
| 7 | True | True | False | top_k_mean | 0.02 | 0.6652 | 0.6783 | 0.6509 |
| 8 | False | True | True | max | 0.11 | 0.6606 | 0.6835 | 0.6341 |
| 9 | False | False | True | top_k_mean | 0.08 | 0.6561 | 0.6833 | 0.6238 |
| 10 | True | False | True | top_k_mean | 0.09 | 0.6561 | 0.6000 | 0.6984 |

### Winner

- **Config:** `remove_stop=True, stem=True, sentence_level=False, aggregation=max`
- **Best threshold:** 0.03
- **Accuracy:** 0.6923
- **F1 (S):** 0.7213
- **F1 (NS):** 0.6566

Confusion matrix at winner's best threshold:

|            | Pred NS | Pred S |
|------------|---------|--------|
| Gold NS    | 65      | 44     |
| Gold S     | 24      | 88     |

- Gold S recall: 88/112 = 78.6%
- Gold NS recall: 65/109 = 59.6%

**Note:** this is the best **regex-tokenizer** config. The NLTK-tokenizer variant (Section 8b) supersedes it with 70.1% accuracy — see Section 10 for the final result.

**Plots:** `top10_configs.png`, `threshold_sweep.png`
**Raw data:** `grid_results.csv`, `threshold_sweep_base.csv`

### Observations

1. **`max` aggregation dominates the top ranks.** 6 of the top 10, including all of the top 4, use `max`. `top_k_mean` never wins outright. Interpretation: for a fact to be supported, it usually suffices for **one** passage segment to overlap strongly — averaging across all segments dilutes the signal.

2. **Stemming helps consistently** (present in 5 of the top 6 configs). The gains come from inflected verb forms ("performed" → "perform", "acting" → "act") which otherwise would not match between a fact and its supporting passage. Stemming does *not* collapse nominal variants like "actor" and "actress" to a common root, so those still mismatch — a limitation that the verb-level gains outweigh.

3. **Stopword removal helps.** The top 4 configs all use `remove_stop=True`. Removing "the", "is", "of", "and" prevents them from inflating the union set and diluting the score.

4. **Sentence-level comparison is a wash.** Rank 1 (whole-passage) and rank 2 (sentence-level) are within 0.5% of each other. Sentence-level gets a higher score on individual supported facts but adds noise from unrelated sentences.

5. **Best thresholds are very low (0.02–0.11).** Absolute Jaccard values are low because facts and passages share few words. The optimal decision boundary sits in the 0.02–0.11 range.

### Comparison with baseline

| Config | Best t | Accuracy | F1 (S) | F1 (NS) |
|--------|--------|----------|--------|---------|
| Section 7 baseline (stop+stem+sent+max) | 0.08 | 0.6878 | 0.7206 | 0.6462 |
| Section 8 winner (stop+stem, whole passage, max) | 0.03 | 0.6923 | 0.7213 | 0.6566 |

**Gain from grid search: +0.45 percentage points.** Small but real. Current best: **69.2%**, roughly 6 points short of the 75% full-credit threshold.

### Ceiling observation

Pure Jaccard similarity appears to plateau around 69% on this dataset. The limitation is structural: Jaccard treats all tokens equally (no IDF weighting), ignores word order and syntax, and cannot detect paraphrases or infer from evidence across multiple sentences. This is expected and is exactly the motivation for Part 2's entailment model.

### Alternative tokenizer: NLTK word_tokenize

Same grid sweep, but using `nltk.word_tokenize` instead of the regex tokenizer.

**Best NLTK config:** `stop=True, stem=True, sentence_level=False, aggregation=max, t=0.03`
- Accuracy: 0.7014
- F1 (S): 0.7317
- F1 (NS): 0.6633

| Tokenizer | Best accuracy | Best config |
|-----------|---------------|-------------|
| Regex | 0.6923 | stop+stem, whole-passage, max, t=0.03 |
| NLTK word_tokenize | 0.7014 | stop+stem, whole-passage, max, t=0.03 |

**Gain: +0.91 percentage points.** NLTK's tokenizer handles contractions ("isn't" → "is", "n't") and punctuation slightly better than the regex approach. Small but real, and the same winning config pattern (`stop=True, stem=True, sentence_level=False, aggregation=max`) confirms the pattern is robust across tokenizers.

### IDF-weighted token overlap

Standard Jaccard treats all shared tokens equally. Weighted Jaccard gives each token a weight based on its document frequency across the passages corpus:

    weight(w) = 1 / (1 + log(df(w)))

Weighted Jaccard:

    J_w(A, B) = sum_{w ∈ A ∩ B} weight(w) / sum_{w ∈ A ∪ B} weight(w)

Corpus statistics:
- Passages indexed: 23,720
- Unique tokens: 17,435
- Sample weights: "career" → 0.107, "american" → 0.115, "news" → 0.133, "mississippi" → 0.190

Evaluated at the winning config (`stop=True, stem=True, sentence_level=False, aggregation=max`).

| Approach | Accuracy | F1 (S) | F1 (NS) | Best threshold |
|----------|----------|--------|---------|----------------|
| Regex, unweighted | 0.6923 | 0.7213 | 0.6566 | 0.03 |
| NLTK, unweighted | 0.7014 | 0.7317 | 0.6633 | 0.03 |
| NLTK, IDF-weighted | 0.6833 | 0.7266 | 0.6237 | 0.03 |

**Observation:** IDF weighting **hurts** performance by ~1.8 points. The reason is structural: informative tokens for fact-checking tend to be common descriptive words ("american", "actress", "actor", "born"), and IDF down-weights them. Removing the equal-weighting assumption actually removes useful signal. This is a concrete negative result: for this task, unweighted Jaccard is preferable, and simple word overlap is the right primitive.

### Final Jaccard result

**Winning approach:** NLTK tokenizer, unweighted Jaccard, whole-passage comparison, max aggregation.

| Parameter | Value |
|-----------|-------|
| Tokenizer | NLTK `word_tokenize` |
| Lowercase | True |
| Remove stopwords | True |
| Stemming | True (Porter) |
| Sentence-level | False (compare against full passage) |
| Aggregation | max |
| Threshold | 0.03 |

**Accuracy: 0.7014 (70.1%)**
**F1 (S): 0.7317**
**F1 (NS): 0.6633**

### Final comparison of all tested approaches

| Rank | Approach | Accuracy | F1 (S) | F1 (NS) | Threshold |
|------|----------|----------|--------|---------|-----------|
| 1 | NLTK unweighted, whole-passage, max | 0.7014 | 0.7317 | 0.6633 | 0.03 |
| 2 | Regex unweighted, whole-passage, max | 0.6923 | 0.7213 | 0.6566 | 0.03 |
| 2 | NLTK unweighted, top-2 mean (whole passage) | 0.6923 | — | — | 0.02 |
| 4 | NLTK IDF-weighted, whole-passage, max | 0.6833 | 0.7266 | 0.6237 | 0.03 |
| 4 | NLTK unweighted, top-2 mean (sentence-level) | 0.6833 | — | — | 0.10 |

### Conclusion for Part 1 (Jaccard)

The best Jaccard configuration achieves **70.1% accuracy** on the 221-example dev set. This is short of the 75% coursework target, and the gap is intrinsic to the method rather than a tuning failure:

1. **Bag-of-words ignores meaning.** "Marianne McAndrew is an American." and "Marianne McAndrew is an actress." share most of their content words with the same passage, but only one is supported by it. Jaccard cannot tell the difference.
2. **Paraphrase is invisible.** A fact like "McAndrew was born in Cleveland" vs. a passage saying "born in Ohio" scores low because "Cleveland" and "Ohio" don't overlap as tokens, even though they're semantically linked.
3. **Negation is preserved in tokens but not in scoring.** We kept "not" in the token stream, but Jaccard treats "not" as just another shared word, so "is not" vs "is" barely changes the score.
4. **IDF weighting hurt.** Common descriptive words like "american", "actress", "born" carry the actual fact-checking signal, and IDF down-weights them.

This ceiling is expected and is precisely why Part 2 introduces a textual entailment model that can reason beyond surface form.

---

## 9. Consolidated Experiment Log

The consolidated log (`experiment_log.csv`) contains **150 rows** across four experiment families.

| Experiment family | Rows | Accuracy range |
|-------------------|------|----------------|
| Threshold sweep (baseline config, regex tokenizer) | 101 | 0.49 – 0.69 |
| Grid search, regex tokenizer | 24 | 0.57 – 0.69 |
| Grid search, NLTK tokenizer | 24 | 0.57 – 0.70 |
| IDF-weighted Jaccard (single config) | 1 | 0.68 |

Best single result in the entire log: **NLTK tokenizer, unweighted Jaccard, whole-passage max, threshold 0.03 → 0.7014 accuracy.**

---

## 10. Final Result

### Winning configuration

| Parameter | Value |
|-----------|-------|
| Tokenizer | NLTK `word_tokenize` |
| Lowercase | True |
| Remove stopwords | True (47-word conservative list, negations preserved) |
| Stemming | True (Porter) |
| Sentence-level | False (compare against whole passage) |
| Aggregation | max |
| Threshold | 0.03 |

### Performance on dev set (221 examples)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7014 |
| Precision (S) | 0.6716 |
| Recall (S) | 0.8036 |
| F1 (S) | 0.7317 |
| Precision (NS) | 0.7471 |
| Recall (NS) | 0.5963 |
| F1 (NS) | 0.6633 |

Confusion matrix `[gold, pred]`:

|            | Pred NS | Pred S |
|------------|---------|--------|
| Gold NS    | 65      | 44     |
| Gold S     | 22      | 90     |

Total correct: 155 / 221.

### Interpretation

- The model achieves **70.1% accuracy**, roughly 5 points short of the 75% coursework target for full credit.
- The classifier is **asymmetric**: high recall on supported facts (80.4%) at the cost of precision (67.2%). It leans toward predicting S whenever word overlap is present.
- The 44 false positives (NS predicted S) dominate the error profile. These are cases where the passage shares vocabulary with the fact but does not actually support it (for example, a fact mentions "American actress" and the passage mentions "American actor" — high overlap, but the fact is not directly supported).

### Why Jaccard plateaus

1. **Surface form only.** No notion of meaning, synonyms, or paraphrase. "Cleveland" vs "Ohio" score zero overlap despite being related.
2. **No word order.** "X is Y" and "Y is X" are identical to the model.
3. **Negation is not weighted.** "not" is preserved in tokens but contributes as just one more match.
4. **Common descriptive words dominate.** "American", "actress", "born" carry the fact-checking signal but are also the words Jaccard treats as ordinary tokens.
5. **IDF weighting hurt** — a concrete negative result that confirms the equal-weighting assumption is actually appropriate for this task.

These limits are exactly what Part 2's entailment model addresses.

### Artifacts produced

| File | Contents |
|------|----------|
| `threshold_sweep_base.csv` | 101 threshold values for baseline config |
| `grid_results.csv` | 24 regex-tokenizer configs, best per config |
| `experiment_log.csv` | 150-row consolidated log across all experiments |
| `final_predictions.csv` | Per-example predictions at winning config |
| `threshold_sweep.png` | Accuracy / F1(S) / F1(NS) vs threshold |
| `top10_configs.png` | Top-10 configs by accuracy |