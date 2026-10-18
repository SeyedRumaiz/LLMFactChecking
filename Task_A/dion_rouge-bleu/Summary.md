# ROUGE and BLEU for Lexical Fact Checking: Experimental Summary

## Abstract

The study investigated lexical evidence matching for binary fact verification using ROUGE and BLEU. A dataset of 221 annotated facts from 31 biography topics was evaluated using retrieved Wikipedia passages. Thirty-two preprocessing configurations and sixteen metric variants were compared. Preprocessing, metric variants and decision thresholds were selected within training biographies using grouped cross-validation. Five outer biography-grouped folds produced classification accuracies of **78.28% for the ROUGE selection procedure** and **75.57% for the BLEU selection procedure**. The final refitted pipelines used Porter stemming, stop-word removal and biography-title-word removal, with different evidence-window lengths. Higher development-tuned accuracies were observed, but those estimates reused tuning labels. The experiments supported lexical matching as an effective baseline without demonstrating the aspirational 85% validation target.

## 1. Objective and scope

The objective was to determine whether retrieved passages support a short factual statement using lexical overlap. The prediction task assigned either supported (`S`) or not supported (`NS`). The study focused on selecting effective preprocessing sequences, evidence units, ROUGE/BLEU variants and classification thresholds while reducing direct tuning leakage.

The final analysis contains two independently optimized method families. ROUGE and BLEU were permitted to select different preprocessing and evidence lengths. No neural entailment model, fact-specific rule, manually specified year correction or named-person lookup was included. The current ROUGE/BLEU study is distinct from the earlier experiments involving additional metric families and supervised combinations.

## 2. Dataset and experimental design

### 2.1 Data preparation

The supplied annotations contained 112 supported facts, 93 not-supported facts and 16 irrelevant facts. Following the supplied evaluator, irrelevant labels were mapped to `NS`, producing 112 supported and 109 not-supported examples. Each fact was associated with the supplied retrieved passages through an exact fact-text match. The resulting dataset contained 221 matched examples across 31 biography topics. Passage retrieval and atomic-fact extraction were supplied components rather than newly trained stages.

### 2.2 Grouped model selection and evaluation

Biography topics defined the validation groups. All facts associated with a biography remained together within a split. Five outer folds were used for evaluation, and three inner grouped folds were used for selecting preprocessing, metric variants and thresholds. A fixed random seed of 42 was used for the outer partition; inner partitions used the recorded fold-specific seeds.

Within an outer round, candidate settings were compared using only the outer training biographies. Thresholds were fitted on inner training examples and evaluated on inner validation examples. The candidate with the highest pooled inner-validation accuracy was selected separately for each family. A threshold was then refitted on the complete outer training set before predicting the excluded biographies. Every example consequently received one outer-fold prediction.

The reported outer accuracies evaluate the **selection procedure**. Different rounds can select different variants and preprocessing settings. The final model fitted on all 221 examples is a deployment refit rather than an independently evaluated model. Selection-side cross-validation and development-tuned estimates are reported separately in [Results.md](Results.md).

## 3. Methods

### 3.1 Preprocessing and evidence construction

Retrieval markup was removed, HTML entities were decoded, Unicode was normalized using NFKC, and text was case-folded before alphanumeric tokenization. Numbers were retained. Negation terms were retained by the fixed function-word list.

Three binary preprocessing choices were evaluated: Porter stemming, stop-word removal and removal of the normalized biography-title tokens. Title-word removal was applied to both the fact and the evidence. Porter stemming provided an established morphological normalization method instead of the earlier custom suffix rules. No lemmatization model was used.

Four evidence constructions were evaluated:

1. Fixed windows of eight cleaned tokens.
2. Windows equal to the cleaned fact length.
3. Windows equal to twice the cleaned fact length.
4. Sentence-like chunks separated by retrieval tags and punctuation boundaries.

Windows advanced by half the window length, rounded down with a minimum stride of one token. Final windows could be shorter. Window length was determined after preprocessing, and duplicate windows were removed. Sentence construction used lightweight splitting rather than a linguistic sentence parser. The three binary choices and four evidence constructions produced **32 configurations**.

### 3.2 ROUGE variants

Six ROUGE variants were considered: ROUGE-1 recall and F1, ROUGE-2 recall and F1, and ROUGE-L recall and F1. The fact was treated as the reference to be covered by the evidence.

ROUGE-1 and ROUGE-2 used clipped matching counts for unigrams and bigrams. Recall divided the matched count by the reference count; precision divided the matched count by the evidence count. F1 combined precision and recall through the harmonic mean. ROUGE-L used longest-common-subsequence length, preserving relative token order while allowing gaps. Token repetitions were retained, distinguishing these calculations from unique-word-set overlap.

### 3.3 BLEU variants

Ten BLEU variants were considered. BLEU-1 was evaluated in both candidate/reference directions. BLEU-2 and BLEU-4 were each evaluated with add-one or epsilon smoothing and in both directions. Forward scoring used the fact as candidate and evidence as reference. Reverse scoring used evidence as candidate and the fact as reference.

The sentence-level implementation combined clipped n-gram precisions through an equally weighted geometric mean and applied the brevity penalty. For candidates shorter than the requested maximum order, only available orders were used. Add-one smoothing adjusted higher-order counts; epsilon smoothing replaced a zero higher-order precision numerator with 0.1. Zero unigram overlap produced score zero. These settings are explicit smoothed sentence adaptations rather than unsmoothed corpus BLEU. BLEU-1 uses unigram matching and does not measure higher-order sequence agreement.

### 3.4 Scoring and classification

For each metric, the maximum score over the retrieved evidence units represented the fact. A score at or above the fitted threshold produced `S`; a lower score produced `NS`. Empty cleaned facts or absent evidence received score zero.

Threshold selection swept the observed score boundaries, grouping tied values together and considering every distinct attainable binary classification. Consequently, thresholds were selected from the training data rather than from an arbitrary coarse grid. Equal-accuracy threshold ties retained the lower threshold. Equal candidate-selection scores retained the first candidate in the recorded search order.

The search comprised **192 ROUGE configuration/variant combinations** and **320 BLEU combinations**, for **512 combinations in total**. The complete comparison record is available in [all_rouge_bleu_comparisons.csv](all_rouge_bleu_comparisons.csv).

## 4. Results

### 4.1 Outer-validation performance

ROUGE correctly classified 173 of 221 examples, yielding **78.28% accuracy**. BLEU correctly classified 167 examples, yielding **75.57% accuracy**. The equally weighted mean was **76.92%**. An always-supported reference would classify 112 examples correctly, corresponding to 50.68% accuracy.

The ROUGE selection procedure exceeded the BLEU procedure by six correctly classified facts, or 2.71 percentage points. No statistical significance test was conducted, so the observed difference should not be interpreted as a universal ranking. These percentages measure classification accuracy after thresholding, not raw ROUGE or BLEU similarity values. [Figure 1](figures/01_accuracy_comparison.png) separates outer-validation accuracy from development-tuned accuracy.

The highest development-tuned scores were 81.90% for ROUGE-1 recall with sentence chunks and 80.54% for reverse BLEU-1 with twice-fact-length windows. Both used stemming, stop-word removal and title-word removal. Those scores were optimized and evaluated on the same labels and therefore provide optimistic search results rather than independent performance estimates.

### 4.2 Final refitted pipelines

Training-side grouped selection on all available facts selected ROUGE-1 recall with eight-token windows and a threshold of 0.3542. BLEU selected reverse BLEU-4 with add-one smoothing, twice-fact-length windows and a threshold of 0.1934. Both final pipelines enabled Porter stemming, stop-word removal and title-word removal.

Final selection did not necessarily reproduce the development-score winner. In particular, reverse BLEU-1 achieved the highest development-tuned BLEU accuracy, whereas reverse BLEU-4 was selected by the training-side grouped comparison. The final thresholds were fitted on all available examples; outer-fold thresholds varied.

Across the five outer rounds, ROUGE-1 recall was selected consistently, with eight-token windows in two rounds and sentence chunks in three. BLEU selected reverse BLEU-1 in four rounds and reverse BLEU-4 with add-one smoothing in one. Twice-fact-length evidence windows were selected in all BLEU rounds. Porter stemming was enabled in four of five rounds for each family. The fold-specific choices are tabulated in [Results.md](Results.md).

## 5. Interpretation of preprocessing and sequence choices

### 5.1 Preprocessing comparisons

The preprocessing comparisons held the final metric and evidence length fixed, changed one preprocessing operation, and refitted thresholds within three training-side grouped folds. The selected ROUGE preprocessing achieved 78.73% selection-CV accuracy. Disabling stemming reduced that estimate to 76.47%; retaining stop words reduced it to 66.52%; retaining title words reduced it to 69.68%.

The corresponding BLEU estimates were 77.38% for the selected preprocessing, 72.85% without stemming, 74.66% when stop words were retained, and 59.28% when title words were retained. Within these controlled comparisons, the largest reduction for ROUGE followed retention of stop words, while the largest reduction for BLEU followed retention of title words. [Figure 2](figures/02_preprocessing_comparison.png) presents these comparisons.

The findings are consistent with morphological normalization improving some inflectional matches and filtering reducing overlap attributable to grammatical words or biography names. These are interpretations of the observed comparisons, not universal causal conclusions. Removing title words or applying stemming can also erase useful distinctions. The comparisons were used for selection and must not be substituted for outer-validation scores.

### 5.2 Evidence-window comparisons

With the final ROUGE variant and other preprocessing fixed, eight-token windows achieved 78.73% selection-CV accuracy. Sentence chunks achieved 76.92%, twice-fact-length windows 75.11%, and fact-length windows 74.21%. Short windows can limit unrelated material while retaining sufficient fact coverage, although maximizing across many windows can also select accidental matches.

For the final BLEU variant, twice-fact-length windows achieved 77.38%, compared with 73.30% for fact-length windows, 71.04% for eight-token windows and 56.11% for sentence chunks. BLEU depends on candidate length, n-gram precision, brevity penalty and scoring direction. Windows scaled to the fact length offered a useful compromise for this dataset under the selected reverse-scoring formulation. [Figure 3](figures/03_evidence_windows.png) shows the comparisons, and [Figure 4](figures/04_pipeline.png) summarizes the final inference pipelines.

### 5.3 Relationship to earlier approaches

Earlier stages investigated word-overlap baselines, shared preprocessing across four metric families, and regularized score combinations. Those experiments established alternative reference results, but used different objectives or evaluation protocols. For example, a development-tuned four-family mean, a single 52-fact biography holdout and a nested grouped estimate over all 221 facts are not interchangeable measurements. The historical appendix in [Results.md](Results.md) preserves the earlier comparisons without treating the highest percentage across stages as the strongest generalization result.

## 6. Limitations

The dataset contains only 31 biography groups and was explored before the final experiments. Grouped nesting reduces direct leakage during fitting but cannot remove the effects of previous exploration. Selection across numerous configurations can remain variable even when an outer evaluation loop is retained. Shared passage content across different biography topics is another possible source of dependence.

Lexical agreement does not establish logical support. Matching words may describe a different person, date, role or relationship, while valid paraphrases may have limited overlap. Stop-word and title-word removal can create new adjacent token sequences, and stemming can collapse distinctions. The sentence splitter is approximate. These limitations affect both interpretation of metric values and transfer to new biographies.

The study did not demonstrate 85% outer-validation accuracy. The tested search does not establish a theoretical ceiling for ROUGE, BLEU or lexical fact checking. Further changes selected using the current outer-fold results would introduce additional model-selection effects. An independently annotated collection of biographies is required for a cleaner final assessment.

## 7. Conclusion

ROUGE and BLEU provided transparent, reproducible lexical baselines above the assignment's 75% accuracy criterion under the recorded outer-validation protocol. The ROUGE selection procedure achieved the higher observed accuracy. Training-side comparisons supported Porter stemming, stop-word removal and title-word removal in the final refits, with eight-token windows for ROUGE and twice-fact-length windows for BLEU. The main conclusions depend on grouped validation rather than the higher development-tuned scores. The final models, complete comparison tables and saved predictions support reproducibility, while a fresh biography dataset remains the appropriate next evaluation step.

## References and experimental records

- Lin, C.-Y. (2004). [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/).
- Papineni, K., Roukos, S., Ward, T. and Zhu, W.-J. (2002). [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/).
- NLTK documentation. [PorterStemmer](https://www.nltk.org/api/nltk.stem.PorterStemmer.html).
- Experimental records: [Results.md](Results.md), `summary.json`, `predictions.csv`, `all_rouge_bleu_comparisons.csv` and `historical_comparisons.csv`.
- Operational handoff: [Notebook_Walkthrough.md](Notebook_Walkthrough.md).
