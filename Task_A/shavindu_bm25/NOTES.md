# BM25 fact checker

Part 1. The metric is BM25.

## Chosen setting

- Query = fact, document = one sentence from the retrieved passages
- Stop words removed, negations kept, Porter stem, page-title tokens removed from the fact
- IDF from the 1,562 unique passages in `data/passages_bm25_ChatGPT_humfacts.jsonl` (labels not used)
- k1 = 1.2, b = 0.25, threshold = 0.197
- 191/221 on the dev set (86.4%)
- 5-fold CV, seed 42, threshold refit each fold: 0.860
- Same thing but holding out whole biographies: 0.859
- Supported class: precision 99/116, recall 99/112, F1 0.868
- Confusion: TP 99, FP 17, FN 13, TN 92

`factcheck.py` still runs Rumaiz's IDF-recall (194/221). This checker is separate. It lives in `bm25_checker.py`.

## Why b = 0.25 and sentences

On sentences with k1 = 1.2:

| b | correct |
|---|---|
| 0 | 189 |
| 0.25 | 191 |
| 0.75 | 188 |
| 1 | 187 |

Whole passages at k1 = 1.2, b = 0.25 also get 191/221. I kept sentences because the supported F1 is 0.868 vs 0.866, and the biography hold-out is 0.859 vs 0.855. That is one or two facts, not a big gap.

Changing k1 on passages with b = 0 (0.6, 0.9, 1.2, 1.5, 2.0) moved the count by at most one fact.

## Preprocessing, one switch off, same k1 and b

| change | correct | CV |
|---|---|---|
| none | 191 | 0.860 |
| keep stop words | 188 | 0.842 |
| no stemming | 185 | 0.833 |
| keep title tokens | 168 | 0.737 |

Title removal is the one that actually moves the result.

## Errors I actually looked at

- "He has worked at Mayo Clinic" scores 0.432 (predicted S, gold NS). The page gives a PhD from the Mayo Clinic. Same words, different relation.
- "The Fischer Research Laboratory was founded in 1936" scores 0.357 (predicted S, gold NS). The page says Fisher Research Laboratory was founded in 1931. Founded / research / laboratory are enough to clear 0.197. The year is ignored.
- "He has since then starred in numerous films" scores 0.161 (predicted NS, gold S). The page names films and does not use "starred" or "numerous".

## Run

From the repo root, with nltk installed:

```bash
python Task_A/shavindu_bm25/bm25_checker.py
```

That reprints the grid. The default arguments of `BM25FactChecker` are the submitted setting.
