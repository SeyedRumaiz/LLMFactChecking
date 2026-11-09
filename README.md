# LLM Fact Checking

CM3604 Deep Learning coursework (RGU / IIT), Group 25.

This project checks whether short facts taken from ChatGPT-generated biographies are supported by retrieved Wikipedia passages. It uses two approaches: word overlap (Part 1) and textual entailment with a pre-trained DeBERTa-v3 NLI model (Part 2). Part 3 is an error analysis of the entailment model, written up in the report.

The data comes from the [FActScore](https://github.com/shmsw25/FActScore) dataset: 221 human-labelled atomic facts from 31 biographies. Each fact is labelled S (supported), NS (not supported) or IR (irrelevant), and IR is counted as NS.

## Results

Both final checkers are in `factcheck.py`.

| Part | Final method | Dev accuracy | 5-fold CV | Target |
|---|---|---|---|---|
| 1. Word overlap | IDF-weighted recall, subject name removed, global IDF, sentence evidence, θ = 0.457 | **194/221 (87.8%)** | 86.9% | ≥ 75% |
| 2. Textual entailment | DeBERTa-v3, two-sentence window premises, max P(E) ≥ 0.325, fact-level pruning at τ = 0.1 | **191/221 (86.4%)** | 85.5% ± 2.3 | ≥ 83% |

Part 3 examines 10 false positives and 10 false negatives of the entailment model (30 errors in total) and groups them into four categories: topical overlap without the claimed relation, vague or general claims, evidence in lists or noisy text, and different wording made worse by pruning.

## Team

| Member | Part 1 | Part 2 |
|---|---|---|
| Mohamed Razik Seyed Rumaiz | TF-IDF cosine and IDF-weighted recall (final Part 1 model) | Window premises, max P(E), IDF-recall pruning (final Part 2 model) and Part 3 error analysis |
| Tamadhi Liyanage | Jaccard similarity with name removal and sliding windows | Jaccard ranking and Jaccard pruning |
| Dion Rasmika | ROUGE-1 recall over 8-token windows | Sentence, pair and passage premises with a ROUGE-1 gate, plus a supplementary error analysis |
| Shavindu Kesara | Normalised BM25 | BM25 gate, then sentence entailment |

## Setup

Python 3.10 and the pinned package versions in `requirements.txt` are required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- The first entailment run downloads `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` from Hugging Face, so it needs internet access once.
- No spaCy model or NLTK data download is needed. `factcheck.py` only uses spaCy's built-in stop-word list and NLTK's Porter stemmer.

## Run

Run from the repository root (the data paths are relative to it):

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

Expected results:

| Mode | Accuracy | Confusion matrix (TP, FN / FP, TN) | Time (CPU) |
|---|---|---|---|
| `word_overlap` | 194/221 = 0.878 | 97, 15 / 12, 97 | about 1 s |
| `entailment` | 191/221 = 0.864 | 94, 18 / 12, 97 | about 3 to 4 min |

Add `--cuda` to run the entailment model on a GPU, if PyTorch with CUDA is installed. On a CPU-only MacBook Air the entailment run peaks at about 1.3 GiB of memory.

`factchecking_main.py` is the supplied driver and has not been modified.

## How the final checkers work

**Part 1: `WordRecallThresholdFactChecker`**

1. Lowercase, remove `<s>` tags and punctuation, keep digits.
2. Remove spaCy stop words but keep negations (*no, not, never, ...*), then apply Porter stemming.
3. Remove the subject's name (from the page title) from the fact, because it appears in almost every passage.
4. Score each sentence by IDF-weighted recall: the share of the fact's IDF weight found in the sentence. IDF is computed once over all 1,562 unique passages.
5. Predict S if the best sentence score is at least θ = 0.457, the middle of the best range [0.455, 0.459].

**Part 2: `EntailmentFactChecker`**

1. Prune: if the Part 1 score of the fact is below τ = 0.1, predict NS without calling the model. This prunes 44 of the 221 facts (17% of premise–fact pairs) and removes 7 false positives.
2. Split passages on `<s>` tags and sentence punctuation, and drop fragments of fewer than three words.
3. Build window premises (previous sentence + current sentence), so that pronoun sentences keep their subject.
4. Run DeBERTa-v3 in batches of 16 under `torch.no_grad()` and take P(entailment) from the softmax.
5. Predict S as soon as any premise reaches P(E) ≥ θ = 0.325 (early exit), otherwise NS.

With pruning and early exit, 3,394 of the 5,417 premise–fact pairs are sent to the model.

## Repository structure

```
.
├── factcheck.py                  # Final Part 1 and Part 2 checkers (submitted)
├── factchecking_main.py          # Supplied driver (unchanged)
├── requirements.txt              # Pinned dependencies (supplied)
├── data/
│   ├── dev_labeled_ChatGPT.jsonl             # 221 labelled facts
│   ├── passages_bm25_ChatGPT_humfacts.jsonl  # Retrieved Wikipedia passages
│   └── FACTSCORE-LICENSE.txt
├── Task_A/                       # Part 1 experiments, one folder per member
│   ├── rumaiz_tf-idf/            # TF-IDF cosine and IDF-weighted recall (tfidf.ipynb)
│   ├── tamadhi_jaccard/          # Jaccard similarity (jaccard_experiments_v2 (2).ipynb, results/)
│   ├── dion_rouge-bleu/          # ROUGE and BLEU (ROUGE_BLEU_Fact_Checking.ipynb)
│   └── shavindu_bm25/            # BM25 (bm25_checker.py)
├── Task_B/                       # Part 2 experiments, one folder per member
│   ├── rumaiz_entailment/        # Final entailment model and Part 3 (entailment.ipynb, entailment_errors.json)
│   ├── tamadhi_entailment/       # Jaccard ranking and pruning (entailment_new.ipynb)
│   ├── dion_entailment/          # Sentence, pair and passage premises (factcheck.py, part1-3.tex)
│   └── shavindu_entailment/      # BM25 gate (run_entailment.py, pair_probs.jsonl)
└── report/
    ├── main.tex                  # Report source (Parts 1 to 3)
    └── figures/                  # Report figures; diagrams/ holds the editable .excalidraw sources
```

## Reproducing the experiments

Open each notebook from inside its own folder, since the notebooks find `data/` relative to their location.

- **Part 1 (final model):** open `Task_A/rumaiz_tf-idf/tfidf.ipynb` and use Restart & Run All. It runs the preprocessing ablation, the name-removal and global-IDF tuning, 5-fold CV, and the threshold range used in `factcheck.py`.
- **Part 2 and Part 3:** open `Task_B/rumaiz_entailment/entailment.ipynb`. The first run scores all 5,417 premise–fact pairs and caches the probabilities in `cache/`. Later runs reuse the cache to compare decision rules, thresholds and pruning, and to produce the error analysis in `entailment_errors.json`.
- **BM25 (Shavindu):** `python Task_A/shavindu_bm25/bm25_checker.py` and `python Task_B/shavindu_entailment/run_entailment.py` can be run from the repository root.

## Report

`report/main.tex` builds the report. Compile it with XeLaTeX (on Overleaf: Menu → Compiler → XeLaTeX), which uses Times New Roman and falls back to TeX Gyre Termes. The `rgu.png` and `iit.png` logos are not in the repository; put them next to `main.tex` or in `report/figures/`.

## Acknowledgements

The dataset and retrieved passages come from FActScore (Min et al., 2023), released under the MIT License (see `data/FACTSCORE-LICENSE.txt`). The entailment model is [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
