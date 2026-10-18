# ROUGE/BLEU Notebook Walkthrough

## 1. Purpose and entry point

**Notebook:** [ROUGE_BLEU_Fact_Checking.ipynb](../ROUGE_BLEU_Fact_Checking.ipynb)

The notebook trains and evaluates two lexical fact-checking procedures: one selected from ROUGE variants and one selected from BLEU variants. Both predict `S` (supported) or `NS` (not supported) from a fact and retrieved passages. Preprocessing, metric variants and thresholds are selected using training biographies.

This document is an operational handoff: execution instructions, cell responsibilities, data contracts, extension points and troubleshooting. Scientific interpretation is documented in [Summary.md](Summary.md); comparison tables are in [Results.md](Results.md).

## 2. Environment and execution

### Requirements

- A Python notebook environment: Jupyter, a notebook-capable editor, or Google Colab.
- NumPy and NLTK. The notebook recommends `numpy==2.0.2` and `nltk==3.9.1`, consistent with the supplied requirements.
- `ipywidgets` for uploads outside Colab. Jupyter supplies IPython display support.
- CPU execution is sufficient. No GPU, transformer model or NLTK corpus download is required.

If dependencies are missing, run the following in a separate notebook cell, then rerun setup:

```python
%pip install numpy==2.0.2 nltk==3.9.1 ipywidgets
```

The notebook imports available packages rather than enforcing installed versions. The recorded reference calculation used Python 3.12.14, NumPy 2.3.5 and NLTK 3.9.1; Python 3.10 compatibility was checked at the syntax level. For an assignment environment, use the specified versions and record the actual versions used for reproduction.

### Execution sequence

1. Open the ROUGE/BLEU notebook and start a fresh kernel.
2. Run **code cell 1** for setup.
3. Run **code cell 2** and select both required JSONL files.
4. Wait for both uploads to finish, then run **code cell 3**.
5. Run **code cells 4–11** in order.
6. Check the results table and the output directory printed by the final cell.

**Upload is a manual boundary.** A single “Run All” action can reach cell 3 before a Jupyter upload finishes. Run through the upload stage manually first. After cell 3 succeeds, the remaining cells can run sequentially without further input.

Uploads are read from memory. No `/content` path, local data folder or mounted drive is assumed. A kernel restart clears uploaded data and fitted settings, so the upload step must be repeated.

## 3. Input data contracts

Exact filenames are required:

| File | Required content |
|---|---|
| `dev_labeled_ChatGPT.jsonl` | Biography topic and human atomic-fact annotations |
| `passages_bm25_ChatGPT_humfacts.jsonl` | Exact fact text and its retrieved passages |

Both files must contain UTF-8 JSON Lines: one JSON object per nonblank line. UTF-8 byte-order marks are accepted.

### Annotation structure

The loader reads `topic`, `annotations`, and `human-atomic-facts`. Each atomic fact needs `text` and `label`. A minimal structural example is:

```json
{"topic":"Example Person","annotations":[{"human-atomic-facts":[{"text":"Example Person is an actor.","label":"S"}]}]}
```

Missing/null annotation lists and missing/null atomic-fact lists are treated as empty. The expected labels are `S`, `NS` and `IR`; `IR` is converted to `NS`. The current loader does not explicitly reject other label values, so label validation is required before using a new dataset.

### Retrieval structure

The loader reads `sent` and `passages`; every passage needs `title` and `text`:

```json
{"sent":"Example Person is an actor.","passages":[{"title":"Example Person","text":"Example Person is an actor."}]}
```

The join is **exact string equality** between atomic-fact `text` and retrieval `sent`. It is not a fuzzy or normalized join. `topic` defines the validation group; the first passage's `title` supplies the title tokens removed during preprocessing.

### Data-quality checks for a new run

- Confirm that intended facts have matching retrieval records. Unmatched annotated facts are currently skipped without a count being printed.
- Check duplicate retrieval `sent` keys. Dictionary construction retains the last record for a duplicate key.
- Validate label values and nonempty biography topics.
- Confirm that passage dictionaries contain string `title` and `text` values, and that titles are consistent within a fact's passages.
- Use at least five distinct biography groups and inspect the class distribution of the resulting folds.
- Keep example ordering fixed once scores and label arrays have been constructed.

These are handoff checks, not additional validation already enforced by the notebook. For the supplied files, the reference load is **221 facts, 31 topics, 112 S and 109 NS** after label conversion.

## 4. Cell-by-cell walkthrough

The notebook has ten numbered sections but **eleven code cells**, because upload and parsing are separate cells. The table below numbers code cells only.

| Code cell | Notebook section | Responsibility | Main objects created |
|---:|---|---|---|
| 1 | Setup | Import dependencies; set seed 42 | `SEED`, imports |
| 2 | Upload data | Open Colab picker or Jupyter upload widget | `uploaded_files`, `uploader` |
| 3 | Read uploaded files | Parse JSONL, join facts to passages, map labels | `examples`, `y`, `all_indices` |
| 4 | Preprocessing | Define text cleaning, 32 configurations and evidence windows | `STOP`, `porter`, `tokenize`, `CONFIGS`, `prepare` |
| 5 | ROUGE and BLEU | Define variants and evidence-scoring functions | `FAMILIES`, `METRICS`, `pair_scores`, `score_example` |
| 6 | Calculate scores | Score every fact under every configuration | `score_bank` |
| 7 | Select settings | Define threshold fitting and grouped inner selection | `tune`, `grouped_folds`, `fit_threshold`, `select_model` |
| 8 | Run validation | Fit on outer training topics and predict excluded topics | `oof`, `fold_choices`, `validation_accuracy` |
| 9 | Results | Compare outer validation with development-tuned maxima | `development_best`, printed table |
| 10 | Final classifiers | Select and refit deployment settings using all facts | `final_settings`, `FactChecker` |
| 11 | Save results | Export JSON summary and paired validation predictions | `OUTPUT`, `summary.json`, `predictions.csv` |

### Cells 4–5: preprocessing and scoring

Cleaning removes tags, decodes entities, applies Unicode NFKC normalization and case-folding, then extracts alphanumeric tokens. Optional stop-word removal occurs before optional Porter stemming. Normalized title tokens are subsequently removed from both fact and passages when enabled. Numbers and negations are retained by the configured stop-word list.

The grid is `2 stemming choices × 2 stop-word choices × 2 title-removal choices × 4 evidence types = 32 configurations`.

`prepare(example, config)` returns a cleaned fact tuple and deduplicated, nonempty evidence tuples. Evidence consists of eight-token windows, fact-length windows, twice-fact-length windows, or approximate sentence chunks. Length is measured after cleaning. Window stride is `max(1, length // 2)`; the last window may be shorter.

ROUGE has six variants; BLEU has ten. BLEU uses explicit sentence-level smoothing, effective order for short candidates and both candidate/reference directions. Reverse BLEU treats evidence as candidate and the fact as reference. `pair_scores()` returns all sixteen raw scores for a fact/evidence pair. `score_example()` takes the maximum separately for each variant across evidence windows. Different variants can therefore obtain their maxima from different windows.

An empty cleaned fact or no evidence yields score zero. This is a scoring convention; classification still depends on the fitted threshold.

### Cell 6: score storage

`score_bank[config_index][metric_name]` is a NumPy array with one value per example, in the same order as `examples` and `y`. Scoring uses only the fact and its supplied passages, so these features can be computed before splitting without fitting on validation labels.

The expensive work is token processing, repeated n-gram comparison and longest-common-subsequence scoring. Runtime depends on passage length, candidate-window count and hardware. This cell prints progress after every eight configurations.

### Cells 7–8: selection and validation

`tune()` searches score boundaries, moving tied scores together. Prediction is `S` when `score >= threshold`. Equal-accuracy threshold ties retain the lower threshold. `select_model()` evaluates each candidate through three inner grouped folds and retains the first candidate when selection accuracies tie. Changing grid or variant order can therefore change tied selections.

`grouped_folds()` uses the global example topics, a local seeded shuffle and round-robin assignment of topics to folds. All facts about a biography remain on one side of a split. Accuracy is pooled over examples, not averaged equally over biography topics.

Cell 8 performs five outer rounds. Each family selects settings using the outer training set, refits a threshold on that set, and predicts only the outer test indices. The saved `oof` arrays are the correct source for validation accuracy and prediction exports. `fold_choices` records the selected variants, configurations and thresholds; outer test topic lists are not exported by this notebook.

### Cells 9–10: interpretation and deployment refit

Cell 9 reports two distinct estimates. **Validation** is the outer-fold estimate of the selection procedure. **Development (tuned)** is the highest score after fitting thresholds and searching configurations on all labels; it is optimistic.

Cell 10 selects final settings through grouped selection on all examples and refits thresholds on all labels. The final all-data classifiers are suitable for subsequent predictions, but their settings are not independently evaluated by rerunning training accuracy. The final variant can differ from the highest development-tuned variant.

## 5. Reference outputs

For the supplied files and recorded implementation:

| Family | Outer validation | Best development-tuned |
|---|---:|---:|
| ROUGE | 78.28% — 173/221 | 81.90% |
| BLEU | 75.57% — 167/221 | 80.54% |

The mean outer accuracy is 76.92%. Expected final settings are:

| Family | Variant | Evidence | Threshold |
|---|---|---|---:|
| ROUGE | ROUGE-1 recall | 8-token windows | 0.3541667 |
| BLEU | BLEU-4 add-one reverse | Twice cleaned fact length | 0.1934090 |

Both final settings enable stemming, stop-word removal and title removal. Different results on changed data are expected; reference values are not assertions to hard-code into training.

## 6. Using a fitted classifier

After cell 10, inference requires only a fact and passages:

```python
checker = FactChecker("ROUGE")  # or "BLEU"
prediction = checker.predict(
    fact="Example Person is an actor.",
    passages=[{"title": "Example Person", "text": "Example Person is an actor."}],
)
print(prediction)  # "S" or "NS"
```

This is a synthetic interface example, not an evaluation example. No gold label or biography topic is needed for prediction.

The JSON export contains thresholds and configuration rather than a pickled estimator. In a later session, run the definitions from code cells 1, 4 and 5, then load the saved settings and use a small inference function:

```python
saved_settings = json.loads(
    Path("rouge_bleu_results/summary.json").read_text(encoding="utf-8")
)["final_settings"]

def predict_saved(family, fact, passages):
    setting = saved_settings[family]
    scores = score_example(
        {"fact": fact, "passages": passages}, setting["config"]
    )
    return "S" if scores[setting["metric"]] >= setting["threshold"] else "NS"
```

The matching preprocessing/scoring code and dependency versions are required alongside the settings. Running the entire training workflow is not necessary for this inference-only route.

## 7. Output locations and reporting

The final cell writes to `Path.cwd() / "rouge_bleu_results"`. The output directory follows the **kernel working directory**, not necessarily the notebook's file location. Existing files with the same names are overwritten.

| Output | Contents |
|---|---|
| `summary.json` | Validation protocol, family accuracies, mean, final settings, development maxima and fold selections |
| `predictions.csv` | One row per matched fact: `fact`, `gold`, `ROUGE`, `BLEU`; predictions are outer-fold predictions |

The notebook does **not** automatically regenerate `Results.md`, `Summary.md`, this walkthrough or the PNGs. Those are separate reporting artifacts in the original project:

1. `compute_all_comparisons.py` calculates the complete 512-row comparison CSV from the project data and companion study script.
2. `compute_comparisons.py` calculates fixed-final-variant preprocessing comparisons.
3. `create_figures.py` renders PNGs from the saved comparison table and experiment summary.
4. `create_reports.py` produces `Results.md` and historical comparison exports. `Summary.md` is an authored narrative and must be reviewed after experiment changes.

These scripts expect the original project structure and on-disk inputs; they are not upload-based notebook cells. Their results must correspond to the same dataset, configuration grid and implementation before being combined with a new notebook run. Figure rendering uses ReportLab and pypdfium2, which are separate from the notebook's training dependencies.

## 8. Changing or extending the study

| Change | Required action |
|---|---|
| Upload different data | Rerun upload/parsing and all downstream cells; inspect joins, labels and topics |
| Modify cleaning, stop words or stemming | Rerun cell 4 and cells 6–11; redefine scoring too if required |
| Modify window lengths or `CONFIGS` | Rebuild `score_bank`, then rerun selection, evaluation and refitting |
| Add or modify a metric | Update `FAMILIES`, `METRICS` and scoring together; rerun cells 5–11 |
| Change folds, seed or threshold selection | Rerun the relevant definitions and cells 8–11; record the revised protocol |
| Update reports or figures | Regenerate matching comparisons and review narrative claims against the new outputs |

`tokenize()` is cached. Changing global preprocessing objects without redefining the function can retain stale tokens; rerun cell 4 or call `tokenize.cache_clear()` before recomputing scores. Never reorder `examples` independently of label arrays or cached score arrays.

For an engineering handoff, retain the notebook version, input-file hashes, dependency versions, seed, configuration grid and exported predictions. The current notebook does not automatically write a complete run manifest. Enlarging the search after examining outer-fold outcomes introduces further model-selection effects; a fresh external dataset remains preferable for a final claim.

## 9. Troubleshooting

| Symptom | Check or resolution |
|---|---|
| Missing NumPy, NLTK or widgets | Install dependencies in the active kernel environment, then rerun setup |
| Upload widget missing | Check Jupyter widget support; Colab uses its own picker |
| Missing-file error in cell 3 | Upload both files with exact expected filenames; wait for completion and rerun parsing |
| No matching facts | Compare atomic-fact `text` with retrieval `sent`; inspect JSONL schema and encoding |
| `NameError` or missing score key | Restart and execute in dependency order; rerun score calculation after changing metric names |
| Slow score calculation | Inspect passage lengths/window counts; wait for progress rather than repeatedly starting parallel runs |
| Empty or unbalanced folds on new data | Inspect distinct topic counts and per-fold class counts before interpreting accuracy |
| Results differ from the reference | Compare inputs, example order, versions, grid/variant order, seed and code changes |
| Output files cannot be found | Inspect `Path.cwd()` and the `OUTPUT` value printed in cell 11 |

## 10. Completion checks

- Upload parsing completed and the fact/topic/label counts were reviewed.
- Every configuration was scored before validation.
- All five outer folds completed for both families.
- Validation and development estimates remained separately labelled.
- Predictions contain one row per matched fact and reproduce the summary accuracies.
- Final settings, scoring code and environment information were retained for inference.
- Any report or figure attached to the run uses the same data and implementation.

The walkthrough documents the existing notebook; no additional model family or experiment is introduced.
