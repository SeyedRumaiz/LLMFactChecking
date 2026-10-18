# ROUGE and BLEU: results and summary

## What was done
We checked whether short facts are supported by the supplied Wikipedia passages. The study uses **221 facts from 31 biographies**, with `IR` treated as `NS`, as required by the evaluator.

We compared **32 preprocessing settings** and **16 ROUGE/BLEU variants**. Each method selected its own preprocessing, variant and threshold using training biographies. Five outer biography groups measured accuracy; three inner groups were used for selection.

## Results

| Method | Grouped validation | Best development-tuned |
|---|---:|---:|
| ROUGE | **78.28%** | 81.90% |
| BLEU | **75.57%** | 80.54% |

**Validation is the main result.** Development scores reuse tuning labels and are optimistic. Neither method demonstrated 85% validation accuracy in this search.

## Final choices

Both pipelines lowercase and clean text, apply Porter stemming, remove stop words and remove biography-title words. Numbers and negations are retained.

| Method | Final variant | Evidence | Threshold |
|---|---|---|---:|
| ROUGE | ROUGE-1 recall | Overlapping 8-token windows | 0.3542 |
| BLEU | BLEU-4, add-one smoothing, reverse | Windows twice the cleaned fact length | 0.1934 |

Reverse BLEU uses evidence as the candidate and the fact as the reference. The highest window score is compared with the threshold to predict `S` or `NS`.

## Why these choices?
The final choices came from training-side grouped comparisons, not hand-written rules for particular facts. Figures 2 and 3 hold the final variant fixed, change one preprocessing operation or evidence length, and refit thresholds within each training fold. They explain selection; they are not additional independent test scores.

Choices were not identical in every outer fold: Porter stemming was selected in 4/5 ROUGE folds and 4/5 BLEU folds. The final refitted model is therefore distinct from the five models used to produce validation predictions.

## Figures

1. [Accuracy comparison](figures/01_accuracy_comparison.png) — validation versus optimistic development scores.
2. [Preprocessing comparison](figures/02_preprocessing_comparison.png) — remove one operation at a time.
3. [Evidence windows](figures/03_evidence_windows.png) — compare window lengths.
4. [Pipeline](figures/04_pipeline.png) — the final processing and decision steps.

## Limit
This small dataset was explored previously. Grouped validation reduces direct tuning leakage, but a fresh set of biographies is still needed for an independent final test. These results do not prove that higher accuracy is impossible.

Files: `summary.json` contains the original experiment results; `predictions.csv` contains outer-fold predictions; `preprocessing_comparisons.csv` contains the training-side comparison values.
