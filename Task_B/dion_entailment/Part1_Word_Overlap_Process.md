# Part 1: Word Overlap — Process Walkthrough

This note records **what was actually done** for CM3604 Part 1 (bag-of-words overlap fact-checking), in the order the work happened. It is a lab log, not a second implementation. The code that the brief asks you to submit lives in `factcheck.py`, class `WordRecallThresholdFactChecker`. The official check is:

```text
python factchecking_main.py --mode word_overlap
```

**Verified result on this machine:** `179/221 = 81.0%` (target in the brief: **at least 75%**).

---

## 0. What the brief asked for

Part 1 is **not** a neural network. It is a lexical overlap classifier:

1. Take a human-annotated atomic fact (a short sentence from a ChatGPT biography).
2. Compare it to Wikipedia passages retrieved by BM25.
3. Compute a **real-valued overlap score**.
4. Apply a **threshold** → predict `S` (supported) or `NS` (not supported).
5. Tune preprocessing and the threshold on the development set.

The brief explicitly allows cosine TF–IDF, Jaccard, precision/recall overlap, ROUGE, BLEU, or anything similar. The feedback grid rewards **justified preprocessing**, **a correct similarity method**, **threshold tuning**, and **comparison of alternatives**.

The driver `factchecking_main.py` must not be edited. It already:

- loads facts and passages
- maps gold label `IR` (irrelevant) → `NS` for scoring
- instantiates `WordRecallThresholdFactChecker()` **with no arguments**
- prints accuracy and per-class P/R/F1

So every modelling choice has to be **frozen as defaults** inside `factcheck.py`.

---



## 1. Problem framing and data sources



### Task

Each example is a triple:


| Field      | Meaning                                                          |
| ---------- | ---------------------------------------------------------------- |
| `fact`     | One atomic proposition, e.g. “Gerhard Fischer is an inventor.”   |
| `passages` | List of `{title, text}` Wikipedia chunks retrieved for that fact |
| `label`    | Human label: `S`, `NS`, or `IR`                                  |


The model must output only `S` or `NS`. Irrelevant gold labels are treated as not supported.

### Files


| File                                        | Role                                              |
| ------------------------------------------- | ------------------------------------------------- |
| `data/dev_labeled_ChatGPT.jsonl`            | ChatGPT bios + **human-atomic-facts** with labels |
| `data/passages_bm25_ChatGPT_humfacts.jsonl` | BM25 Wikipedia passages keyed by fact string      |
| `factchecking_main.py`                      | Loader, evaluation loop (do not change)           |
| `factcheck.py`                              | Our classifier                                    |


The jsonl files originally sat at the repo root. They were linked into `data/` because that is where the driver looks.

### How examples are built (no extra cleaning of labels)

`read_fact_examples` walks each biography → each annotation → each `human-atomic-facts` item. A fact is kept only if:

- annotations are not `null`
- `human-atomic-facts` is not `null`
- the fact string exists in the passage lookup

Missing facts are printed and skipped. We did **not** hand-edit labels or drop classes.

**Official run size:** **221** labelled facts.

From the official confusion matrix:

```text
          pred S    pred NS
gold S       94        18      → 112 supported
gold NS      24        85      → 109 not supported (includes IR)
Accuracy  179/221 = 81.0%
```

The classes are nearly balanced, so a dummy “always S” or “always NS” would sit near 50%, not 75%.

---



## 2. Exploratory data analysis (EDA)

EDA was done **before** freezing a scorer, by reading the jsonl structure and a sample of passages. Findings that changed the design:

### 2.1 Facts are short; passages are long and messy

Facts are one clause. Passages are BM25 hits with Wikipedia markup, often wrapped in `<s>…</s>` sentence tags, and they mix lead sections, career paragraphs, and boilerplate.

Example (abbreviated) from the passage file:

```text
<s>Lanny Flaherty Lanny Flaherty (born July 27, 1942) is an American actor.</s>
<s>Career. He has given his most memorable performances in …</s>
```

**Implication:** scoring the **whole passage as one bag of words** dilutes the supporting span. A short fact can match a few tokens while the rest of the chunk is unrelated. That is why we later scored **local windows**, not the full document.

### 2.2 The page title is repeated everywhere

Almost every fact starts with the person’s name. Almost every passage starts with the same name. Unigram overlap on the **title tokens alone** looks like support even when the rest of the fact is wrong.

Example:

- Fact: “Marianne McAndrew is a singer.” (gold **NS**)
- Title: `Marianne McAndrew`
- Passage: “Marianne McAndrew is an American actress…”

Name overlap is high; the occupation “singer” may never appear. If we count title words as evidence, we get **false positives**.

**Implication:** try **stripping title tokens** from both the fact and the passage after normalisation.

### 2.3 Gold `IR` is rare but real

Some facts are off-topic (e.g. “Cleveland is in Ohio.” while the page is about an actress). The driver already maps `IR` → `NS`. We did not train a three-way classifier.

### 2.4 Support is local, not global

Supported facts usually have a **short contiguous span** that restates the proposition (“is an inventor”, “appeared in A Bronx Tale”). Unsupported facts often share only function words or the name.

**Implication:** a **recall** of fact tokens inside a small window is a better match to “is this claim covered?” than Jaccard over the whole page (Jaccard punishes long passages).

### 2.5 Zero lexical overlap still exists

After title stripping, **8 gold-S facts** have overlap **0**. Those cannot be saved by bag-of-words. They are a ceiling on Part 1, and a reason Part 2 (entailment) exists.

### 2.6 What we did *not* do

- No plots are required by the brief.
- No external Wikipedia dump.
- No class rebalancing.
- No dropping of hard biographies.
- IDF was **not** fitted on the whole labelled set for the submitted model (see scoring). The submitted method is unigram recall, which needs no IDF.

---



## 3. Data cleaning

Cleaning here means **making text comparable**, not rewriting the dataset.


| Step                                         | What                                 | Why                                                                   |
| -------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| Skip null annotations / missing passage keys | Driver behaviour                     | Avoid crashing on incomplete rows                                     |
| Map `IR` → `NS`                              | Driver evaluation                    | Brief: predict only S vs NS                                           |
| `html.unescape`                              | `&` → `&`, etc.                      | Wikipedia/HTML entities otherwise become extra tokens                 |
| Strip `<…>` tags                             | Regex `<[^>]+>` → space              | Remove `<s>`, `</s>`, leftover markup so they are not “words”         |
| Unicode NFKC                                 | `unicodedata.normalize`              | Fold compatibility characters (fancy quotes, full-width letters)      |
| Lowercasing                                  | `casefold()`                         | “American” vs “american” should match                                 |
| Collapse nothing else                        | We do **not** delete digits or names | Years (`1931`, `1993`) and proper names are often the factual payload |


We did **not**:

- spell-correct ChatGPT or Wikipedia
- translate
- remove numbers
- remove negations (`not`, `never` are **not** in the stop list on purpose)

---



## 4. Preprocessing (tokenization, stops, stemming, title)

This is the pipeline inside `_normalize` / `_content_tokens` in `factcheck.py`.

### 4.1 Tokenisation

```text
re.findall(r"[^\W_]+", cleaned)
```

That keeps letters and digits, drops punctuation and underscores. It is a **deliberate simple tokenizer**:

- no spaCy (optional in the starter, not required for Part 1)
- no WordPiece (that is Part 2)
- contractions split naturally (`don't` → `don`, `t`) which is acceptable for overlap

**Why not whitespace split?** Wikipedia has punctuation glued to names (`"Hello, Dolly!"`). Regex word tokens are more stable.

### 4.2 Stopword removal

Fixed list (not learned from this dataset):

`a an the is are was were be been being am to of in on at by for from with as and or but that this these those it its his her their he she they who which has have had do does did`

**Why remove them?** Almost every fact is of the form “X **is** a **Y**”. The copula and articles match everywhere and do not indicate support.

**Why a fixed list?** Fitting stopwords on 221 labelled facts would leak the development set into preprocessing. A generic function-word list is independent of this homework dump.

**Why keep numbers and negations?** “1931” vs “1936” is often the whole difference between S and NS. “not” can flip support.

### 4.3 Porter stemming (on)

NLTK `PorterStemmer`: `inventor` / `inventing` / `invented` → `invent`.

**Why on?** Biographies paraphrase (“invented a metal detector” vs “inventor of the metal detector”). Stemming increases recall of the fact’s content words.

**Cost:** over-stemming can merge unrelated words. Nested comparison with vs without Porter still favoured **stemming on** together with title stripping and windows.

### 4.4 Title-token stripping (on)

After stemming, tokens that appear in the Wikipedia **page title** are removed from both the fact and the passage.

**Why?** See EDA 2.2. Name overlap is a trap.

**Cost:** if the fact *is only the name plus a copula*, the fact bag can become empty and the score is 0 → `NS`. That is preferable to claiming support from the title alone.

Worked example:

```text
Fact:    "Gerhard Fischer is an inventor."
Title:   "Gerhard Fischer (inventor)"   → tokens gerhard, fischer, inventor
Stops:   is, an
After title strip: fact bag may be empty or nearly empty if "inventor" is in the title.

Fact:    "Gerhard Fischer invented the first handheld metal detector."
After strip: invent, first, handheld, metal, detector
Those must be found in a passage window — that is real evidence.
```



### 4.5 Frozen preprocessing vector

From the grouped-CV search over 16 configs:


| Choice              | Frozen value                 |
| ------------------- | ---------------------------- |
| Porter stemming     | **True**                     |
| Remove title tokens | **True**                     |
| Evidence unit       | **8-token window**, stride 4 |


Other evidence units tried in the study: whole-sentence splits, windows of length = fact length, windows of length = 2× fact length. The 8-token window was selected because support is local and 8 content tokens is roughly one short clause after stops are removed.

---



## 5. Scoring method (the “model”)

There is **no trained neural net** in Part 1. The model is a score plus a threshold.

### 5.1 Evidence units

For each passage, content tokens are cut into overlapping windows of **8** tokens, step **4**. Short passages stay as a single unit.

### 5.2 Unigram recall (submitted)

Let F be the set of content tokens in the fact, W a window.


\mathrm{recall}(F, W) = \frac{|F \cap W|}{|F|}
\quad (0 \text{ if } F \text{ is empty})


The example score is the **maximum** recall over all windows of all passages.

**Predict** `S` **if score ≥ 0.35416666666666663, else** `NS`**.**

**Why recall, not precision?** We care whether the **claim is covered** by some span of Wikipedia, not whether the span is a short extractive summary. Precision would punish windows that contain extra true biography tokens.

**Why max over windows?** The ~~b~~rief’s intuition: a supported fact should overlap **some** supporting span highly, not average over the whole article.

**Why not cosine TF–IDF in the submitted file?** It was implemented and compared. Recall was simpler, matched the class name `WordRecallThresholdFactChecker`, and on the official driver it reached **81.0%**, above 75%. TF–IDF needs an IDF estimate; we used only **per-example retrieval-local IDF** in the study (no labels), but recall needs none of that at submission time.

### 5.3 Alternatives that were compared (same preprocessing grid)


| Family             | Metrics                         | Nested grouped-CV accuracy (study) |
| ------------------ | ------------------------------- | ---------------------------------- |
| TF–IDF cosine      | count TF–IDF, binary TF–IDF     | ~76.9%                             |
| Jaccard            | |F \cap W| / |F \cup W|         | ~76.9%                             |
| Precision / recall | recall, F1, F2                  | ~77.4%                             |
| ROUGE / BLEU       | ROUGE-1 recall, ROUGE-L, BLEU-2 | ~78.3%                             |


All four families cleared 75% under nested CV. **ROUGE-1 recall** and **unigram recall** are almost the same object on this tokenisation, which is why they tied on the full-set fit (**179/221**).

A logistic **combination** of metrics plus coverage features reached ~80.5% nested, but the brief asks for a discrete word-overlap score, not an extra classifier. The submitted method stays a **single recall score**.

### 5.4 Threshold tuning

The brief allows tuning the threshold on this development set (there is no hidden test).

Procedure used in the study:

1. Sort examples by score.
2. Sweep the cut: everything ≥ t → `S`.
3. Pick t that maximises accuracy.
4. Midpoint between adjacent scores so the cut is not tied to one row.

Frozen cut: **t = 0.35416666666666663** (just over one third of the fact’s content tokens must appear in some 8-token window).

Interpretation: a 3-content-word fact needs at least one of those words in a window to be borderline; a 6-word fact needs about two.

**Honesty check:** fitting t on all 221 labels is optimistic. Nested **grouped** CV (group = biography/topic, so facts about one person do not leak across folds) still put recall/ROUGE in the **~77–78%** band, still above 75%. Quote **81.0%** as the official-driver number the marker will see; quote nested CV if you discuss generalisation in the write-up.

---



## 6. End-to-end inference (what happens at predict time)

For one fact and its passages:

1. Normalise title → title token set.
2. Normalise fact → drop stops, stem, drop title tokens → set F.
3. If F is empty → score `0` → `NS`.
4. For each passage, same normalisation, slide windows of 8 / stride 4.
5. Score = max unigram recall.
6. If score ≥ `0.354…` → `S`, else `NS`.

No Wikipedia API, no GPU, no fitted vocabulary file. The constructor takes no arguments, as the driver requires.

---



## 7. Official evaluation

Command:

```text
python factchecking_main.py --mode word_overlap
```


| Quantity       | Value                 |
| -------------- | --------------------- |
| Examples       | 221                   |
| Accuracy       | **179 / 221 = 81.0%** |
| Precision `S`  | 94 / 118 ≈ 79.7%      |
| Recall `S`     | 94 / 112 ≈ 83.9%      |
| F1 `S`         | ≈ 81.7%               |
| Precision `NS` | 85 / 103 ≈ 82.5%      |
| Recall `NS`    | 85 / 109 ≈ 78.0%      |
| F1 `NS`        | ≈ 80.2%               |


Errors that bag-of-words cannot fix (preview for Part 3 / Part 2):

- **False positives:** name or occupation words appear in a window, but the sentence does not actually support the claim (wrong year, wrong film, co-occurrence).
- **False negatives:** paraphrase or inference with little token overlap after title strip (the 8 gold-S zeros); support split across a long sentence so no 8-gram window covers enough fact tokens.

---



## 8. Why these choices vs the Grade A grid


| Grid expectation         | What we did                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| Tokenisation             | Regex word tokens after Unicode/HTML/tag clean                                                           |
| Stemming / lemmatisation | Porter stemming, compared on/off                                                                         |
| Stop-words               | Fixed function-word list; negations and numerals kept                                                    |
| Similarity method        | Unigram recall; Jaccard, TF–IDF, ROUGE/BLEU compared                                                     |
| Threshold                | Accuracy-maximising sweep, frozen in code                                                                |
| Preprocessing search     | 16 configs: stem × title-strip × 4 evidence units                                                        |
| Validation               | Grouped nested CV in the study; official number is the driver on the same dev set (allowed by the brief) |
| Reproducible             | All choices hardcoded in `factcheck.py`                                                                  |


---



## 9. How this feeds Part 2 (not Part 1 submission)

Word overlap is reused as a **signal**, not as the Part 2 decision:

- Very aggressive “if overlap < 0.12 then skip DeBERTa” would drop those **8 gold-S zeros**, so we did **not** discard whole facts that way.
- Sentence splitting of `<s>` regions *is* used in Part 2, because NLI wants premises, not 8-token bags.

Part 1 stops at the recall threshold.

---



## 10. Files to look at


| Path                                                               | What it is                                                                                 |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `factcheck.py` → `overlap_score`, `WordRecallThresholdFactChecker` | Submitted Part 1 method                                                                    |
| `factchecking_main.py`                                             | Data load + accuracy (unchanged)                                                           |
| `test_factcheck.py`                                                | Small overlap tests (covered fact → `S`, no overlap → `NS`, title not counted as evidence) |
| `CW.pdf`                                                           | Requirements and feedback grid                                                             |


Unit check:

```text
python -m unittest test_factcheck.py -v
```

---



## 11. One-page algorithm card

```text
clean(text):
    unescape HTML, strip tags, NFKC, casefold
    tokens = alphanumeric words
    drop stopwords, Porter-stem

score(fact, passages):
    T = tokens(title)
    F = tokens(fact) minus T
    if F empty: return 0
    best = 0
    for each passage:
        W = tokens(passage) minus T
        for each 8-token window of W (stride 4):
            best = max(best, |F ∩ window| / |F|)
    return best

predict:
    S if score >= 0.35416666666666663 else NS
```

That is the full Part 1 process: inspect the FActScore dump, clean markup, choose lexical preprocessing that kills name-only matches, score **local coverage of the fact**, tune one threshold, freeze it in `factcheck.py`.