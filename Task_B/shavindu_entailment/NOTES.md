# Sentence entailment with a BM25 gate

Shavindu. Part 2. Same model as the rest of the group: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Column 0 of the softmax is entailment, which is the order already checked for this checkpoint.

Premises are single sentences (split on `<s>`, then on `.!?`, drop fragments under 3 words). Not windows, and not sentence+pair+passage.

## Submitted rule

- Score every sentence with the Part 1 BM25 checker (k1 = 1.2, b = 0.25, title tokens removed).
- If the best sentence is under 0.05, predict NS and do not call the model. That drops 47 facts and 977 sentences.
- Otherwise the score is the max entailment probability. Predict S at 0.33 or above.

186/221. TP 86, FP 9, FN 26, TN 100. Supported precision 86/95 = 0.905, recall 86/112 = 0.768. 0.325 gives the same 186. From 0.30 to 0.35 the count stays between 184 and 186.

The same cut with no gate is 181/221. A gate of 0.10 at the same cut is 184/221. Using only the single best BM25 sentence is 168/221, so the lexical winner is often the wrong sentence to show the model.

5-fold refit of the cut (seed 42, gate fixed): 0.814. Two folds pick a cut near 0.05. If the search is limited to cuts from 0.20 to 0.50, the same folds score 0.833.

`factcheck.py` still uses the window checker (191/221). This one loses pronoun sentences because the name is in the previous line.

## Files

- `pair_probs.jsonl` — 5,412 sentences, fields `fact_idx`, `premise`, `bm25`, `p_e`, `p_n`, `p_c`
- `run_entailment.py` — rebuilds that file if it is missing, then prints a grid

The grid printer will also show cuts that are only a BM25 threshold in disguise if a later edit lets the probability cut go to 0. A cut of 0 with a BM25 gate of 0.20 reproduces the Part 1 confusion matrix (99/17/13/92). That is not an entailment result. The submitted cut is 0.33.
