# Part 1 Results: Word-Overlap Fact Checking

## The short version

- **Final model:** IDF-weighted recall, passage level, threshold **θ = 0.605**
- **Accuracy:** **181 / 221 = 81.9%** (target: 75%)
- **Held-out estimate (5-fold CV):** **79.6%**, still well above target
- **Runs with:** `python factchecking_main.py --mode word_overlap`, in under 1 second

---

## 1. The data

| | Count |
|---|---|
| Biographies (written by ChatGPT) | 31 |
| Atomic facts | 221 |
| Supported (S) | 112 |
| Not Supported (NS) | 93 |
| Irrelevant (IR, counted as NS) | 16 |
| **Binary split** | **112 S vs 109 NS** |

> **In simple terms:** each fact is a short claim like *"Marianne McAndrew was born in Cleveland."* We have to decide whether the Wikipedia passages back it up. Guessing "S" every time would only get about 51%.

---

## 2. How the model works

1. **Clean the text:** lowercase, remove punctuation, remove common words (*the, was, also*), and stem words (*acting -> act*).
2. **Score the fact:** what share of the fact's important words appear in the passage?
3. **Decide:** if the score ≥ 0.605, predict **Supported**; otherwise **Not Supported**.

> **In simple terms:** if most of the fact's *meaningful* words show up in the Wikipedia passage, it's probably true.

### Why "important" words?

Each word is weighted by **IDF** (inverse document frequency). Rare words count more than common ones.

| Word in the fact | How common in the passages | Weight |
|---|---|---|
| *McAndrew* (the subject's name) | appears everywhere | low |
| *born* | fairly common | medium |
| *Cleveland* | rare | **high** |

So matching *"Cleveland"* counts far more than matching the person's name, which is always going to be there anyway.

---

## 3. Experiment results (ablation study)

We changed one design choice at a time to see what actually helps. Each row has its own best threshold.

| Rank | Configuration | θ | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 🥇 1 | **IDF-recall, passage level** | 0.605 | **0.819** | 0.883 | 0.741 | 0.806 |
| 2 | IDF-recall, sentence level | 0.424 | 0.801 | 0.770 | 0.866 | 0.815 |
| 3 | Cosine, unigrams, stopwords + stemming | 0.270 | 0.760 | 0.736 | 0.821 | 0.776 |
| 4 | Cosine, unigrams, stopwords only | 0.260 | 0.733 | 0.715 | 0.786 | 0.749 |
| 5 | Cosine, passage level, uni + bigrams | 0.040 | 0.724 | 0.658 | 0.946 | 0.777 |
| 6 | Cosine, uni + bigrams *(baseline)* | 0.125 | 0.720 | 0.669 | 0.884 | 0.762 |
| 7 | Cosine, unigrams, no preprocessing | 0.288 | 0.701 | 0.730 | 0.652 | 0.689 |
| 8 | Cosine, uni + bigrams, no preprocessing | 0.144 | 0.692 | 0.667 | 0.786 | 0.721 |

### What we learned

| Finding | Effect | Simple explanation |
|---|---|---|
| **Recall beats cosine** | +6 points | Cosine punishes long passages for having extra words. Recall only asks *"is the fact covered?"* |
| **Stopwords + stemming help** | +6 points (0.701 -> 0.760) | Common words add noise; stemming lets *"acted"* match *"acting"*. |
| **Bigrams hurt** | −4 points (0.760 -> 0.720) | Facts are only 5–10 words, so exact two-word matches are rare and noisy. |
| **Passage level helps recall** | +2 points (0.801 -> 0.819) | Facts are often spread over two sentences (*"She appeared in…"*, with the name in the sentence before). |

---

## 4. Is the threshold reliable?

### Threshold sensitivity

| θ | Accuracy |
|---|---|
| 0.500 | 0.783 |
| 0.580 | 0.805 |
| 0.600 | 0.805 |
| **0.605** | **0.819** |
| 0.610 | 0.815 |
| 0.650 | 0.810 |
| 0.700 | 0.783 |

- Accuracy stays **≥ 80%** for any θ between **0.56 and 0.68**.
- Accuracy stays **≥ 75%** for any θ between **0.40 and 1.0**.

> **In simple terms:** the model isn't fragile. You don't need to hit 0.605 exactly; anything around 0.6 works. Cosine, by contrast, had a sharp peak: moving its threshold by ±0.02 cost 2–6 points.

### 5-fold cross-validation

We picked θ using 80% of the facts, then tested on the other 20%, five times over.

| Fold | Chosen θ | Held-out accuracy |
|---|---|---|
| 1 | 0.604 | 0.778 |
| 2 | 0.601 | 0.727 |
| 3 | 0.605 | 0.773 |
| 4 | 0.657 | 0.818 |
| 5 | 0.605 | 0.886 |
| **Mean** | **≈ 0.61** | **0.796** |

> **In simple terms:** even on facts it wasn't tuned on, the model scores about 80%. The threshold also kept landing near 0.6, which suggests it's a real pattern, not luck.

---

## 5. Final model breakdown

|  | Predicted S | Predicted NS |
|---|---|---|
| **Actually S** | ✅ 83 | ❌ 29 (missed) |
| **Actually NS** | ❌ 11 (false alarm) | ✅ 98 |

| Metric | Value | Meaning |
|---|---|---|
| Precision (S) | 0.88 | When it says "supported", it's right 88% of the time |
| Recall (S) | 0.74 | It finds 74% of the truly supported facts |
| Precision (NS) | 0.77 | When it says "not supported", it's right 77% of the time |
| Recall (NS) | 0.90 | It catches 90% of the false or unsupported facts |

> **In simple terms:** the model is **cautious**. It rarely calls something true when it isn't (11 false alarms), but it misses some true facts (29) when Wikipedia words them differently.

---

## 6. Where word overlap fails

| Problem | Example | Result |
|---|---|---|
| **Paraphrase** | Fact: *"actress"*; Wikipedia: *"appeared in films"* | Missed (false negative) |
| **Wrong detail, same words** | Fact: *"born November 21, 1942"*; Wikipedia: *November 22, 1942* | False alarm (false positive) |
| **Order ignored** | *"born in Ohio, died in California"* vs the reverse | Can't tell them apart |
| **Vague claims** | *"Protect the Boss is popular"* | Wikipedia rarely says "popular" |
| **General knowledge (IR)** | *"Cleveland is in Ohio."* | Words match, but it's labelled NS |

> **In simple terms:** word overlap checks whether the *words* match, not whether the *meaning* matches. That's what Part 2 (textual entailment with DeBERTa) is for.

---

## 7. Files

| File | What's in it |
|---|---|
| `factcheck.py` | Final model: `WordRecallThresholdFactChecker` |
| `Task_A/rumaiz_tf-idf/tfidf.ipynb` | All experiments, ablation and plots |
| `Task_A/rumaiz_tf-idf/ablation_results.csv` | Ablation table |
| `Task_A/rumaiz_tf-idf/figures/` | Accuracy-by-config, accuracy-vs-threshold, score distribution |
| `Task_A/rumaiz_tf-idf/TFIDF.md` | Detailed method explanation with formulas |
