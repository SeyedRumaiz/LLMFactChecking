# Part 1: Word-Overlap Fact Checking (Detailed Results)

---

## 1. Executive summary

| | Value |
|---|---|
| Final method | IDF-weighted recall, passage-level evidence, spaCy stopwords (negations kept), Porter stemming |
| Decision threshold | θ = 0.605 (grid search, step 0.001) |
| **Dev accuracy** | **181 / 221 = 81.90%** (target ≥ 75%) |
| 95% bootstrap CI | [76.5%, 86.9%] |
| 5-fold CV held-out accuracy | 79.6% ± 5.3 |
| Threshold-free separability (ROC-AUC) | 0.881 (vs 0.803 for best TF-IDF cosine) |
| Runtime (221 facts, CPU) | ≈ 0.6 s |

**Key findings**

1. Replacing symmetric TF-IDF cosine with asymmetric IDF-weighted recall is the single most effective change: +4.1 pts with preprocessing held fixed, +5.9 pts best-vs-best.
2. Recall-based scores separate the classes better *regardless of threshold* (AUC 0.881 vs 0.803), so the gain doesn't depend on a lucky θ.
3. The IDF-recall threshold sits on a broad plateau (≥ 80% for θ ∈ [0.56, 0.68]), while the cosine optimum is a sharp spike.
4. With n = 221, the accuracy gain over the best cosine model is **not statistically significant** at α = 0.05 (McNemar exact p = 0.105). The AUC gap and CV results still consistently favour IDF-recall.
5. The model is **precision-oriented** (S precision 0.88, S recall 0.74). Most errors are paraphrases, vague claims and cases where every word is present but the claimed relation differs.

---

## 2. Experimental setup

### 2.1 Data

| Item | Count |
|---|---|
| ChatGPT biographies | 31 |
| Human-annotated atomic facts | 221 |
| Gold S | 112 (50.7%) |
| Gold NS | 93 (42.1%) |
| Gold IR (mapped to NS) | 16 (7.2%) |
| Retrieved passages per fact | 2–5 (BM25, from the subject's Wikipedia page) |

IR is mapped to NS to match `factchecking_main.py` (`"NS" if label == "IR"`). The binary task is balanced (112 vs 109), so the majority-class baseline is **50.7%**.

### 2.2 Evaluation protocol

Applied identically to every configuration:

| Step | Setting |
|---|---|
| Example loading | same fact → passage matching as `factchecking_main.py` (exact sentence text; 221/221 matched) |
| Decision rule | predict S iff score ≥ θ |
| Threshold search | exhaustive grid θ ∈ [0, 1], step 0.001 (1,001 values), tuned per configuration |
| Objective | maximise accuracy; S-class F1 breaks ties |
| Reported metrics | accuracy, S-class precision / recall / F1, confusion matrix |
| Robustness | threshold sweep, 5-fold CV (seed 42), 10,000-sample bootstrap CI, McNemar test |

All components (TF-IDF, cosine, IDF, metrics) are implemented from scratch with Python dictionaries and NumPy. There is no scikit-learn dependency, so `factcheck.py` runs using only `requirements.txt`.

---

## 3. Methods compared

### 3.1 Preprocessing pipeline

`lowercase → strip <s> tags → punctuation → space → whitespace split → stopword removal → Porter stemming`

| Choice | Decision | Rationale |
|---|---|---|
| Stopword list | spaCy English (326 words) **minus** `no, not, nor, never, none, n't` | Removes high-frequency function words; keeping negations prevents "X did **not** win" matching "X won" |
| Stemming | Porter | Collapses morphological variants (*acting/acted → act*) |
| Digits | kept | Dates and years are often the deciding detail in biographies |
| N-grams | unigrams (bigrams tested) | See §4.2 |

### 3.2 Scoring functions

Let *F* be the fact's term set and *U* an evidence unit's term set. IDF is computed over a **local corpus**, [fact] + that fact's evidence units, with smoothing:

$$\text{idf}(t) = \log\frac{N+1}{\text{df}(t)+1} + 1$$

| Method | Score per unit | Fact score |
|---|---|---|
| TF-IDF cosine | $\cos(\vec{q}, \vec{u})$, with TF = 1 + log(count), L2-normalised | max over units |
| **IDF-weighted recall** | $\dfrac{\sum_{t \in F \cap U} \text{idf}(t)}{\sum_{t \in F} \text{idf}(t)}$ | max over units |

**Why recall?** Cosine is symmetric: every unmatched evidence term lowers the score. Facts have 1–8 content terms, while evidence units are long, so supporting passages are penalised for their length. Recall is asymmetric and measures only how much of the fact is covered, which is the quantity that matters for verification. IDF weighting means rare, informative terms (*Cleveland*, *1942*) dominate, while the subject's name, which appears in every unit and so has low IDF, contributes little.

---

## 4. Results

### 4.1 Full ablation

| # | Metric | Unit | N-grams | Stop | Stem | θ | Acc | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **IDF-recall** | **passage** | 1 | ✓ | ✓ | 0.605 | **0.8190** | 0.883 | 0.741 | 0.806 |
| 2 | IDF-recall | sentence | 1 | ✓ | ✓ | 0.424 | 0.8009 | 0.770 | 0.866 | 0.815 |
| 3 | Cosine | sentence | 1 | ✓ | ✓ | 0.270 | 0.7602 | 0.736 | 0.821 | 0.776 |
| 4 | Cosine | sentence | 1 | ✓ | ✗ | 0.260 | 0.7330 | 0.715 | 0.786 | 0.749 |
| 5 | Cosine | passage | 1+2 | ✓ | ✓ | 0.040 | 0.7240 | 0.658 | 0.946 | 0.777 |
| 6 | Cosine *(baseline)* | sentence | 1+2 | ✓ | ✓ | 0.125 | 0.7195 | 0.669 | 0.884 | 0.762 |
| 7 | Cosine | sentence | 1 | ✗ | ✗ | 0.288 | 0.7014 | 0.730 | 0.652 | 0.689 |
| 8 | Cosine | sentence | 1+2 | ✗ | ✗ | 0.144 | 0.6923 | 0.667 | 0.786 | 0.721 |

P, R and F1 are for the S class. Source: `ablation_results.csv`.

### 4.2 Controlled comparisons (one factor changed)

| Factor | From → To | Configs | Δ accuracy |
|---|---|---|---|
| **Metric** | cosine → IDF-recall (sentence, uni, stop + stem) | 3 → 2 | **+4.07 pts** |
| Stemming | off → on (cosine, sentence, uni, stop) | 4 → 3 | +2.72 pts |
| Stopwords + stemming | off → on (cosine, sentence, uni) | 7 → 3 | +5.88 pts |
| Stopwords + stemming | off → on (cosine, sentence, uni + bi) | 8 → 6 | +2.72 pts |
| Bigrams | uni → uni + bi (cosine, sentence, stop + stem) | 3 → 6 | **−4.07 pts** |
| Evidence unit | sentence → passage (IDF-recall) | 2 → 1 | +1.81 pts |
| Evidence unit | sentence → passage (cosine, uni + bi) | 6 → 5 | +0.45 pts, but θ collapses to 0.04 |

**Interpretation**

- **Bigrams hurt** because facts are short. An exact two-word match is rare, so bigram features add mostly zero-overlap dimensions that dilute cosine.
- **Passage level helps recall but not cosine.** A larger unit can only *increase* recall (more chances to cover each term), and it captures facts whose terms are spread over adjacent sentences, e.g. a pronoun subject plus a predicate. For cosine, longer units increase the vector norm and depress every score. The optimal θ drops to 0.04, so the classifier degenerates towards "always S" (recall 0.95, precision 0.66).

### 4.3 Threshold-free separability (ROC-AUC)

AUC measures how well a score ranks S facts above NS facts, independent of θ.

| Score | ROC-AUC | Mean score (S) | Mean score (NS) | Median (S) | Median (NS) |
|---|---|---|---|---|---|
| **IDF-recall, passage** | **0.881** | 0.809 | 0.358 | 1.000 | 0.340 |
| IDF-recall, sentence | 0.878 | 0.713 | 0.311 | 0.690 | 0.303 |
| Cosine, sentence (best) | 0.803 | 0.395 | 0.214 | 0.382 | 0.227 |

IDF-recall's advantage is **structural, not a threshold artefact**: its AUC is 0.08 higher. The score distribution is strongly bimodal:

| | Gold S | Gold NS |
|---|---|---|
| Score = 1.0 (every content term found) | **65** | 5 |
| Score = 0.0 (no content term found) | 1 | **18** |

### 4.4 Threshold analysis

**Sensitivity (final model)**

| θ | Accuracy | Precision (S) | Recall (S) |
|---|---|---|---|
| 0.400 | 0.738 | 0.675 | 0.929 |
| 0.500 | 0.783 | 0.750 | 0.857 |
| 0.580 | 0.805 | n/a | n/a |
| 0.600 | 0.805 | n/a | n/a |
| **0.605** | **0.819** | **0.883** | **0.741** |
| 0.610 | 0.815 | n/a | n/a |
| 0.700 | 0.783 | 0.900 | 0.643 |
| 0.800 | 0.783 | 0.932 | 0.616 |
| 1.000 | 0.765 | 0.929 | 0.580 |

- Accuracy ≥ 80% for θ ∈ **[0.561, 0.678]**; ≥ 75% for θ ∈ **[0.404, 1.000]**.
- Even θ = 1.0 ("every content word must appear") scores 76.5%, because 65/112 S facts are fully covered.
- θ trades precision for recall smoothly; 0.605 is the accuracy-optimal balance point.

**Contrast with cosine:** the best cosine model peaks at 0.760 (θ = 0.270) and falls to 0.738 / 0.724 at θ ± 0.02. Its optimum is a narrow spike (see `figures/accuracy_vs_threshold.png`).

**The 0.600 → 0.605 step:** the +1.4 pt jump comes from 3 facts with scores in (0.600, 0.605]. This fine-grained gain is dev-set fitting and shouldn't be over-interpreted; the plateau and CV figures are the reliable estimates.

### 4.5 Generalisation

**5-fold cross-validation** (θ tuned on 4/5, tested on the held-out 1/5; seed 42):

| Fold | θ (tuned on train) | Train acc | Held-out acc | n |
|---|---|---|---|---|
| 1 | 0.604 | 0.824 | 0.778 | 45 |
| 2 | 0.601 | 0.836 | 0.727 | 44 |
| 3 | 0.605 | 0.831 | 0.773 | 44 |
| 4 | 0.657 | 0.814 | 0.818 | 44 |
| 5 | 0.605 | 0.802 | 0.886 | 44 |
| **Mean** | 0.601 – 0.657 | 0.821 | **0.796 ± 0.053** | |

- The optimism gap (train − held-out) is about 2.5 pts, which is small.
- θ lands in [0.601, 0.605] in 4/5 folds, so the decision boundary is stable.

**Bootstrap 95% CI** (10,000 resamples of the 221 facts): accuracy **[76.5%, 86.9%]**. The lower bound is still above the 75% target.

### 4.6 Statistical comparison with the best cosine model

| | IDF-recall correct | IDF-recall wrong |
|---|---|---|
| **Cosine correct** | 147 | 21 |
| **Cosine wrong** | **34** | 19 |

- **McNemar exact test:** b = 34, c = 21, **p = 0.105**.
- **Bootstrap 95% CI of the accuracy difference:** [−0.5, +12.7] pts.

**Interpretation:** IDF-recall fixes 34 of cosine's errors and introduces 21 new ones (net +13). With only 221 facts this isn't significant at α = 0.05, so the accuracy gain alone shouldn't be overstated. The case for IDF-recall rests on the **converging evidence**: higher AUC (0.881 vs 0.803), a robust threshold plateau, and CV accuracy above the target.

---

## 5. Final model

### 5.1 Confusion matrix (θ = 0.605)

|  | Predicted S | Predicted NS | Total |
|---|---|---|---|
| **Gold S** | 83 (TP) | 29 (FN) | 112 |
| **Gold NS / IR** | 11 (FP) | 98 (TN) | 109 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| S | 0.883 | 0.741 | 0.806 |
| NS | 0.772 | 0.899 | 0.831 |

Matches `python factchecking_main.py --mode word_overlap` exactly (181/221).

### 5.2 Accuracy by original gold label

| Gold label | n | Correct | Accuracy | Mean score |
|---|---|---|---|---|
| S | 112 | 83 | 74.1% | 0.809 |
| NS | 93 | 83 | 89.2% | 0.367 |
| IR | 16 | 15 | **93.8%** | 0.310 |

IR facts are handled well: wrong-person claims (e.g. *Fernandinho* in *Fernando*'s bio) share few terms with the subject's page. The single IR error is general knowledge, *"Cleveland is in Ohio."*, which scores 1.0 because the passage contains "Cleveland, Ohio".

### 5.3 Accuracy by fact length (distinct content terms)

| Content terms | n | Accuracy |
|---|---|---|
| 1–2 | 27 | 85.2% |
| 3 | 74 | 79.7% |
| 4 | 50 | 76.0% |
| ≥ 5 | 70 | 87.1% |

Mid-length facts (3–4 terms) are hardest: one missing term moves the score by 25–33%, which pushes borderline facts across θ. Long facts are more robust, since one missing term costs less, and very short facts tend to be all-or-nothing.

### 5.4 Accuracy by biography

- 11 of 31 biographies are classified **perfectly**.
- Hardest: *Jidenna* (1/5), *Jonathan Haagensen* (4/9), *Daniel Carvalho* (3/5), *Ji Sung* (11/18), *Hibo Wardere* (5/8).
- The difficult biographies contain many **vague or stylistic claims** (*"music is a fusion of hip-hop"*, *"incorporates African elements into his style"*, *"is popular"*) and **career narratives paraphrased** from Wikipedia.

---

## 6. Error analysis

### 6.1 False positives (11): predicted S, gold NS / IR

| Score | Gold | Fact | Observed pattern |
|---|---|---|---|
| 1.000 | NS | Marianne McAndrew is an American. | all terms present; attribute not asserted by the evidence |
| 1.000 | NS | Trivisonno worked at WKNR. | all terms present; relation differs |
| 1.000 | NS | He served as a sniper. | all terms present; relation differs |
| 1.000 | NS | Kerser is from Sydney. | all terms present; relation differs |
| 1.000 | IR | Cleveland is in Ohio. | general knowledge, not about the subject |
| 0.770 | NS | Jidenna's music is a fusion of hip-hop. | genre terms co-occur, claim not stated |
| 0.770 | NS | Jidenna's music is a fusion of R&B. | genre terms co-occur, claim not stated |
| 0.717 | NS | The Fischer Research Laboratory was founded in 1936. | entity matches; date detail missing or wrong |
| 0.657 | NS | Botak Chin is also known as Chin Peng. | names co-occur; identity claim wrong |
| 0.643 | NS | Nisar Ahmad Kakru was Kashmiri. | place term present in another context |
| 0.606 | NS | Salazar scored six goals for the Peru national team. | entity matches; number not supported |

**Pattern:** 5 of 11 false positives have a perfect score of 1.0. Every content word is present, but the evidence doesn't assert the claimed **relation**. Bag-of-words matching can't detect this at any threshold.

### 6.2 False negatives (29): predicted NS, gold S

| Pattern | Count | Examples (score) |
|---|---|---|
| **Near-threshold** (0.50 ≤ score < 0.605) | 13 | *"He has worked at Continuum Analytics."* (0.603); *"Katsu has written novels."* (0.537) |
| **Vague / evaluative claims** ("popular") | 5 | *"Kill Me, Heal Me." is popular.* (0.581); *"The films he has starred in have been popular."* (0.525) |
| **Pronoun subject** (He / She / They) | 11 | *"He moved to Dutch club FC Groningen."* (0.299); *"They divorced in 1964."* (0.000) |
| **Data noise** | 2 | *"Jonathan Haagensen**is** an actor."* (missing space → "haagensenis" is one token, 0.588); *-"Protect the Boss." is popular.* |
| **Paraphrase** | most of the remainder | *"Haagensen later transitioned to television."* (0.175); *"Rio de Janeiro is in Brazil."* (0.220) |

The categories overlap: a fact can be both pronoun-subject and near-threshold.

**Pattern:** false negatives are dominated by **lexical mismatch**. Wikipedia states the same fact in different words (*"transitioned to television"* vs *"appeared in TV series"*), or the claim is evaluative and never stated verbatim. 13 of 29 fall just below θ, which is consistent with the precision-oriented operating point.

---

## 7. Efficiency

| Stage | Time (221 facts) |
|---|---|
| IDF-recall scoring (passage level) | ≈ 0.6 s |
| TF-IDF cosine scoring (sentence level) | ≈ 0.6 s |
| Full grid search (1,001 θ) | < 1 s |

That is roughly **3 ms per fact**, with no GPU, no model download and no external API. This makes IDF-recall well suited as the **pruning filter** for the Part 2 entailment model.

---

## 8. Limitations and threats to validity

| Issue | Impact | Mitigation |
|---|---|---|
| θ tuned and evaluated on the same 221 facts | reported 81.9% is optimistic | 5-fold CV (79.6%), plateau analysis, bootstrap CI |
| Small dataset (221 facts, 31 biographies) | wide CIs; cosine vs recall gap not significant (p = 0.105) | report AUC and CV alongside accuracy |
| Bag-of-words ignores relations | 5 FPs with score 1.0 | requires semantic modelling → Part 2 |
| No paraphrase or synonym handling | many FNs | entailment model (Part 2) |
| Local IDF corpus (per fact) | IDF estimates from few documents are noisy | smoothing; down-weights the subject's name as intended |
| Annotation noise | e.g. "Haagensenis", leading "-" | reported, not corrected, to stay faithful to the data |

---

## 9. Reproducibility

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python factchecking_main.py --mode word_overlap        # → Accuracy: 181/221 = 0.8190
```

| Artefact | Location |
|---|---|
| Experiments, ablation, figures | `Task_A/rumaiz_tf-idf/tfidf.ipynb` (Kernel → Restart & Run All) |
| Ablation table | `Task_A/rumaiz_tf-idf/ablation_results.csv` |
| Figures | `Task_A/rumaiz_tf-idf/figures/` (ablation_accuracy, accuracy_vs_threshold, score_distribution, confusion_matrix) |
| Final classifier | `factcheck.py → WordRecallThresholdFactChecker` (θ = 0.605) |
| Seed | 42 (CV split, bootstrap) |

> §4.3 (AUC), §4.6 (McNemar and bootstrap), and §5.2–5.4 (breakdowns by label, length and biography) were computed with the same scoring code, outside the notebook. Add the corresponding cells to the notebook if these figures are quoted in the report.

---

## 10. References

- Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP.
- Sparck Jones, K. (1972). *A statistical interpretation of term specificity and its application in retrieval.* Journal of Documentation, 28(1).
- Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* Information Processing & Management, 24(5).
- Porter, M. F. (1980). *An algorithm for suffix stripping.* Program, 14(3).
- McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika, 12(2).
