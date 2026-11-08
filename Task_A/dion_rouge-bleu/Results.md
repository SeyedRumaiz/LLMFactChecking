# Results: ROUGE and BLEU

Evaluation labels: **outer validation** = five biography-grouped outer folds with three inner selection folds; **selection CV** = three grouped folds used to choose settings; **development** = thresholds fitted and evaluated on the same labels. These estimates are not interchangeable.

## 1. Main comparison

| Approach | Evaluation | Correct / total | Accuracy |
|---|---|---|---|
| Always supported reference | Fixed prediction | 112 / 221 | 50.68% |
| ROUGE selection procedure | Outer validation | 173 / 221 | 78.28% |
| BLEU selection procedure | Outer validation | 167 / 221 | 75.57% |
| ROUGE/BLEU mean | Outer validation | — | 76.92% |
| ROUGE: ROUGE-1 recall | Best development-tuned | 181 / 221 | 81.90% |
| BLEU: BLEU-1 reverse | Best development-tuned | 178 / 221 | 80.54% |

## 2. Final refitted pipelines

| Family | Selected variant | Porter | Stop-word removal | Title-word removal | Evidence | Threshold |
|---|---|---|---|---|---|---|
| ROUGE | ROUGE-1 recall | On | On | On | 8 tokens | 0.354167 |
| BLEU | BLEU-4 add-one reverse | On | On | On | 2 x fact length | 0.193409 |

Final refits use all available facts. Outer-validation accuracy evaluates the fold-by-fold selection procedure, not an independent test of the final all-data refit.

## 3. Preprocessing comparison

Final variant and evidence length fixed; only the named operation changes. Thresholds are refitted within each selection fold.

| Family | Preprocessing change | Selection-CV accuracy | Change from selected (percentage points) |
|---|---|---|---|
| ROUGE | Selected preprocessing | 78.73% | +0.00 |
| ROUGE | No Porter stemming | 76.47% | -2.26 |
| ROUGE | Keep stop words | 66.52% | -12.22 |
| ROUGE | Keep title words | 69.68% | -9.05 |
| BLEU | Selected preprocessing | 77.38% | +0.00 |
| BLEU | No Porter stemming | 72.85% | -4.52 |
| BLEU | Keep stop words | 74.66% | -2.71 |
| BLEU | Keep title words | 59.28% | -18.10 |

## 4. Evidence-sequence comparison

Final variant, stemming, stop-word removal and title-word removal fixed; thresholds refitted within each selection fold.

| Family | Evidence sequence | Selection-CV accuracy | Selected final evidence |
|---|---|---|---|
| ROUGE | 8 tokens | 78.73% | Yes |
| ROUGE | Fact length | 74.21% | No |
| ROUGE | 2 x fact length | 75.11% | No |
| ROUGE | Sentence chunks | 76.92% | No |
| BLEU | 8 tokens | 71.04% | No |
| BLEU | Fact length | 73.30% | No |
| BLEU | 2 x fact length | 77.38% | Yes |
| BLEU | Sentence chunks | 56.11% | No |

## 5. Metric-variant comparison

Each row selects the highest selection-CV preprocessing setting for that variant. Development accuracy in the final column belongs to that same setting, rather than a separate development winner.

Pipeline codes: **P** = Porter stemming; **S** = stop-word removal; **T** = title-word removal; 1 = on, 0 = off.

| Family | Metric variant | Best selection-CV pipeline | Selection CV | Development, same pipeline |
|---|---|---|---|---|
| ROUGE | ROUGE-1 recall | P1 S1 T1; 8 tokens | 78.73% | 81.00% |
| ROUGE | ROUGE-1 F1 | P1 S1 T1; Fact length | 76.47% | 79.19% |
| ROUGE | ROUGE-2 recall | P0 S1 T0; Sentence chunks | 72.85% | 73.30% |
| ROUGE | ROUGE-2 F1 | P0 S1 T0; Fact length | 70.59% | 72.40% |
| ROUGE | ROUGE-L recall | P1 S1 T1; Sentence chunks | 78.73% | 80.09% |
| ROUGE | ROUGE-L F1 | P1 S1 T1; 2 x fact length | 77.38% | 80.54% |
| BLEU | BLEU-1 | P1 S1 T1; 2 x fact length | 75.11% | 79.64% |
| BLEU | BLEU-1 reverse | P1 S1 T1; 2 x fact length | 75.57% | 80.54% |
| BLEU | BLEU-2 add-one | P0 S1 T1; Fact length | 72.40% | 76.47% |
| BLEU | BLEU-2 add-one reverse | P1 S0 T1; 2 x fact length | 75.57% | 77.83% |
| BLEU | BLEU-2 epsilon | P1 S1 T1; Fact length | 74.21% | 76.47% |
| BLEU | BLEU-2 epsilon reverse | P1 S1 T1; 8 tokens | 71.49% | 71.95% |
| BLEU | BLEU-4 add-one | P1 S1 T1; 2 x fact length | 75.11% | 78.28% |
| BLEU | BLEU-4 add-one reverse | P1 S1 T1; 2 x fact length | 77.38% | 77.83% |
| BLEU | BLEU-4 epsilon | P1 S1 T1; 2 x fact length | 76.02% | 77.38% |
| BLEU | BLEU-4 epsilon reverse | P1 S1 T1; 2 x fact length | 74.21% | 77.83% |

## 6. All preprocessing settings

For each family and setting, the best variant is selected by selection-CV accuracy. Complete results for all **512** combinations, including both accuracy estimates and fitted thresholds, are recorded in [all_rouge_bleu_comparisons.csv](all_rouge_bleu_comparisons.csv).

### ROUGE

| ID | Porter | Remove stop words | Remove title words | Evidence | Best variant | Selection CV | Development, same variant |
|---|---|---|---|---|---|---|---|
| C01 | Off | Off | Off | 8 tokens | ROUGE-2 recall | 69.68% | 71.49% |
| C02 | Off | Off | Off | Fact length | ROUGE-1 recall | 65.61% | 69.68% |
| C03 | Off | Off | Off | 2 x fact length | ROUGE-2 recall | 69.68% | 71.04% |
| C04 | Off | Off | Off | Sentence chunks | ROUGE-1 recall | 68.33% | 73.30% |
| C05 | Off | Off | On | 8 tokens | ROUGE-1 F1 | 67.42% | 72.40% |
| C06 | Off | Off | On | Fact length | ROUGE-1 F1 | 73.30% | 74.21% |
| C07 | Off | Off | On | 2 x fact length | ROUGE-1 F1 | 68.78% | 73.76% |
| C08 | Off | Off | On | Sentence chunks | ROUGE-1 recall | 68.78% | 76.47% |
| C09 | Off | On | Off | 8 tokens | ROUGE-1 F1 | 72.40% | 72.40% |
| C10 | Off | On | Off | Fact length | ROUGE-2 recall | 71.95% | 71.95% |
| C11 | Off | On | Off | 2 x fact length | ROUGE-2 recall | 71.95% | 71.95% |
| C12 | Off | On | Off | Sentence chunks | ROUGE-2 recall | 72.85% | 73.30% |
| C13 | Off | On | On | 8 tokens | ROUGE-1 recall | 76.47% | 78.28% |
| C14 | Off | On | On | Fact length | ROUGE-L F1 | 75.11% | 76.92% |
| C15 | Off | On | On | 2 x fact length | ROUGE-1 recall | 77.38% | 79.19% |
| C16 | Off | On | On | Sentence chunks | ROUGE-1 recall | 77.38% | 79.64% |
| C17 | On | Off | Off | 8 tokens | ROUGE-2 recall | 69.68% | 71.04% |
| C18 | On | Off | Off | Fact length | ROUGE-2 recall | 64.71% | 70.59% |
| C19 | On | Off | Off | 2 x fact length | ROUGE-L recall | 65.61% | 68.78% |
| C20 | On | Off | Off | Sentence chunks | ROUGE-1 recall | 68.33% | 73.30% |
| C21 | On | Off | On | 8 tokens | ROUGE-1 F1 | 73.30% | 73.30% |
| C22 | On | Off | On | Fact length | ROUGE-1 F1 | 73.76% | 74.21% |
| C23 | On | Off | On | 2 x fact length | ROUGE-1 F1 | 70.14% | 75.57% |
| C24 | On | Off | On | Sentence chunks | ROUGE-1 recall | 75.11% | 77.38% |
| C25 | On | On | Off | 8 tokens | ROUGE-1 F1 | 73.30% | 74.21% |
| C26 | On | On | Off | Fact length | ROUGE-2 recall | 71.95% | 71.95% |
| C27 | On | On | Off | 2 x fact length | ROUGE-2 recall | 71.95% | 71.95% |
| C28 | On | On | Off | Sentence chunks | ROUGE-2 recall | 72.85% | 73.30% |
| C29 | On | On | On | 8 tokens | ROUGE-1 recall | 78.73% | 81.00% |
| C30 | On | On | On | Fact length | ROUGE-1 F1 | 76.47% | 79.19% |
| C31 | On | On | On | 2 x fact length | ROUGE-L recall | 78.28% | 80.54% |
| C32 | On | On | On | Sentence chunks | ROUGE-L recall | 78.73% | 80.09% |

### BLEU

| ID | Porter | Remove stop words | Remove title words | Evidence | Best variant | Selection CV | Development, same variant |
|---|---|---|---|---|---|---|---|
| C01 | Off | Off | Off | 8 tokens | BLEU-1 reverse | 68.78% | 69.23% |
| C02 | Off | Off | Off | Fact length | BLEU-4 add-one | 66.97% | 70.14% |
| C03 | Off | Off | Off | 2 x fact length | BLEU-4 add-one reverse | 74.21% | 74.21% |
| C04 | Off | Off | Off | Sentence chunks | BLEU-4 epsilon reverse | 54.75% | 66.52% |
| C05 | Off | Off | On | 8 tokens | BLEU-1 reverse | 66.97% | 67.87% |
| C06 | Off | Off | On | Fact length | BLEU-1 | 72.40% | 73.30% |
| C07 | Off | Off | On | 2 x fact length | BLEU-1 reverse | 75.11% | 75.11% |
| C08 | Off | Off | On | Sentence chunks | BLEU-2 add-one reverse | 65.16% | 66.97% |
| C09 | Off | On | Off | 8 tokens | BLEU-4 add-one reverse | 63.80% | 66.52% |
| C10 | Off | On | Off | Fact length | BLEU-4 epsilon reverse | 70.59% | 72.40% |
| C11 | Off | On | Off | 2 x fact length | BLEU-1 reverse | 66.52% | 72.85% |
| C12 | Off | On | Off | Sentence chunks | BLEU-2 epsilon reverse | 56.11% | 69.23% |
| C13 | Off | On | On | 8 tokens | BLEU-1 reverse | 63.35% | 70.59% |
| C14 | Off | On | On | Fact length | BLEU-2 epsilon | 73.30% | 74.66% |
| C15 | Off | On | On | 2 x fact length | BLEU-2 add-one reverse | 74.66% | 77.38% |
| C16 | Off | On | On | Sentence chunks | BLEU-1 | 65.16% | 66.06% |
| C17 | On | Off | Off | 8 tokens | BLEU-2 add-one reverse | 69.23% | 69.68% |
| C18 | On | Off | Off | Fact length | BLEU-2 epsilon | 66.97% | 71.04% |
| C19 | On | Off | Off | 2 x fact length | BLEU-4 add-one reverse | 73.30% | 73.30% |
| C20 | On | Off | Off | Sentence chunks | BLEU-4 epsilon reverse | 54.30% | 66.52% |
| C21 | On | Off | On | 8 tokens | BLEU-4 epsilon reverse | 68.78% | 69.68% |
| C22 | On | Off | On | Fact length | BLEU-1 | 69.23% | 72.85% |
| C23 | On | Off | On | 2 x fact length | BLEU-2 add-one reverse | 75.57% | 77.83% |
| C24 | On | Off | On | Sentence chunks | BLEU-2 add-one reverse | 66.06% | 67.42% |
| C25 | On | On | Off | 8 tokens | BLEU-1 reverse | 67.42% | 68.78% |
| C26 | On | On | Off | Fact length | BLEU-4 epsilon reverse | 71.49% | 72.40% |
| C27 | On | On | Off | 2 x fact length | BLEU-2 add-one | 71.49% | 71.95% |
| C28 | On | On | Off | Sentence chunks | BLEU-2 epsilon reverse | 65.16% | 70.14% |
| C29 | On | On | On | 8 tokens | BLEU-1 reverse | 71.49% | 72.40% |
| C30 | On | On | On | Fact length | BLEU-1 | 74.21% | 78.73% |
| C31 | On | On | On | 2 x fact length | BLEU-4 add-one reverse | 77.38% | 77.83% |
| C32 | On | On | On | Sentence chunks | BLEU-1 reverse | 61.54% | 69.68% |

## 7. Fold-by-fold selections

| Outer fold | Family | Variant | Pipeline | Threshold |
|---|---|---|---|---|
| 1 | ROUGE | ROUGE-1 recall | P1 S1 T1; 8 tokens | 0.354167 |
| 1 | BLEU | BLEU-1 reverse | P0 S1 T1; 2 x fact length | 0.236111 |
| 2 | ROUGE | ROUGE-1 recall | P1 S1 T1; 8 tokens | 0.366667 |
| 2 | BLEU | BLEU-1 reverse | P1 S1 T1; 2 x fact length | 0.236111 |
| 3 | ROUGE | ROUGE-1 recall | P1 S1 T1; Sentence chunks | 0.414286 |
| 3 | BLEU | BLEU-1 reverse | P1 S1 T1; 2 x fact length | 0.236111 |
| 4 | ROUGE | ROUGE-1 recall | P1 S1 T1; Sentence chunks | 0.387500 |
| 4 | BLEU | BLEU-1 reverse | P1 S1 T1; 2 x fact length | 0.183333 |
| 5 | ROUGE | ROUGE-1 recall | P0 S1 T1; Sentence chunks | 0.414286 |
| 5 | BLEU | BLEU-4 add-one reverse | P1 S1 T1; 2 x fact length | 0.193409 |

## 8. Earlier experimental approaches

These rows preserve the earlier stages of the work. Different samples, objectives and validation protocols prevent direct ranking against the current ROUGE/BLEU-only results.

| Stage | Approach | Evaluation | Accuracy |
|---|---|---|---|
| Initial word overlap | Best word recall | Development-tuned | 80.09% |
| Initial word overlap | Train-selected word overlap | Single 52-fact biography holdout | 84.62% |
| Shared four-family preprocessing | TF-IDF cosine | Development-tuned | 78.73% |
| Shared four-family preprocessing | Jaccard | Development-tuned | 78.73% |
| Shared four-family preprocessing | Precision/recall | Development-tuned | 80.54% |
| Shared four-family preprocessing | ROUGE/BLEU | Development-tuned | 80.09% |
| Shared four-family preprocessing | Four-family mean | Development-tuned | 79.52% |
| Shared four-family preprocessing | Four-family mean | Single 52-fact biography holdout | 74.04% |
| Restricted four-family search | TF-IDF cosine | Nested biography-grouped validation | 76.92% |
| Restricted four-family search | Jaccard | Nested biography-grouped validation | 76.92% |
| Restricted four-family search | Precision/recall | Nested biography-grouped validation | 77.38% |
| Restricted four-family search | ROUGE/BLEU | Nested biography-grouped validation | 78.28% |
| Restricted four-family search | Four-family mean | Nested biography-grouped validation | 77.38% |
| Restricted four-family search | Regularized four-score combination | Nested biography-grouped validation | 79.19% |
| Restricted four-family search | Combination + generic coverage | Nested biography-grouped validation | 80.54% |

### Initial word-overlap preprocessing and scoring experiments

| Stop-word removal | Aggregation | Score | Fitted threshold | Development accuracy |
|---|---|---|---|---|
| On | union | recall | 0.732143 | 80.09% |
| Off | union | recall | 0.809091 | 79.19% |
| On | max | recall | 0.633333 | 77.83% |
| Off | max | recall | 0.720779 | 76.92% |
| On | max | cosine | 0.118574 | 76.02% |
| Off | max | cosine | 0.199277 | 73.30% |
| On | max | jaccard | 0.029269 | 72.40% |
| On | union | cosine | 0.082417 | 71.04% |
| Off | union | cosine | 0.142334 | 65.61% |
| On | union | jaccard | 0.004717 | 63.35% |
| Off | max | jaccard | 0.028986 | 60.18% |
| Off | union | jaccard | 0.027701 | 54.30% |

Full earlier-stage comparisons: [historical_comparisons.csv](historical_comparisons.csv). The original four-family tables are also retained in [part1_four_family_search.csv](../../Part1/part1_four_family_search.csv).

## 9. Figures

| Figure | Comparison |
|---|---|
| [1. Accuracy](figures/01_accuracy_comparison.png) | Outer validation versus development-tuned scores |
| [2. Preprocessing](figures/02_preprocessing_comparison.png) | One-operation-at-a-time selection-CV comparisons |
| [3. Evidence windows](figures/03_evidence_windows.png) | Four evidence sequence lengths |
| [4. Pipeline](figures/04_pipeline.png) | Final refitted processing and threshold steps |

## 10. Result sources

- `summary.json`: current outer-validation results, final settings and fold choices.
- `predictions.csv`: 221 paired outer-fold predictions.
- `all_rouge_bleu_comparisons.csv`: all 512 current settings/variant combinations.
- `preprocessing_comparisons.csv`: 64 fixed-final-variant preprocessing comparisons.
- `historical_comparisons.csv`: earlier approaches and available complete search records.
