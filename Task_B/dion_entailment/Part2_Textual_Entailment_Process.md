# Part 2: Textual Entailment — Process Walkthrough

This note records **what was actually done** for CM3604 Part 2 (textual entailment fact-checking), in the same order as the Part 1 log. It is a lab log, not a second implementation. The code the brief asks you to submit lives in `factcheck.py`, classes `EntailmentModel` and `EntailmentFactChecker`. The official check is:

```text
python factchecking_main.py --mode entailment --cuda
```

**Verified result on this machine:** `187/221 = 84.6%` (target in the brief: **at least 83%**).

Part 1 (word overlap) on the same 221 facts was **81.0%**. Entailment is meant to go **beyond surface form** and pick up paraphrase / inference that bags of words miss.

---



## 0. What the brief asked for

Part 2 **is** a neural method, but we do **not** train a model from scratch. We use a **pre-trained** DeBERTa-v3 base already fine-tuned on **MNLI, FEVER, and ANLI**.

For each fact:

1. Treat Wikipedia text as the **premise**.
2. Treat the atomic fact as the **hypothesis**.
3. The model returns three classes: **entailment**, **neutral**, **contradiction**.
4. Map that three-way output to binary `S` **/** `NS`.
5. Optionally **prune** with the Part 1 overlap score so we do not run DeBERTa on hopeless examples.
6. Must not run out of GPU/CPU memory.

The brief allows two mapping families:


| Strategy      | How a fact becomes `S`                                                    |
| ------------- | ------------------------------------------------------------------------- |
| **Discrete**  | If **any part of any passage** is predicted entailment                    |
| **Threshold** | Take the **max** P(\text{entailment}) over parts; `S` if that max ≥ a cut |


The feedback grid also wants:

- a **justified** 3-class → 2-class rule, with alternatives tried
- **sentence splitting** and noise removal
- **word-overlap pruning** (efficiency and/or accuracy)
- reliable execution without OOM
- experimental analysis of those choices

The driver `factchecking_main.py` must not be edited. It already:

- loads the same 221 facts as Part 1
- maps gold `IR` → `NS`
- downloads `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- wraps it in `EntailmentModel` and `EntailmentFactChecker`

We implement **logit → decision** and **how to cut passages into premises**.

---



## 1. Problem framing and data sources



### Task (same examples, new decision)

Same triples as Part 1: `fact`, `passages`, gold `S` / `NS` / `IR`. Same **221** facts after the driver filters missing rows.

NLI is **pair classification**, not document classification. Feeding a 1,200-character BM25 chunk as one premise:

- hits the **512-token** tokenizer cap and truncates the supporting sentence
- mixes many unrelated clauses, which MNLI/FEVER models were not trained to treat as one premise

So Part 2 is mostly **how we build premises**, then **how we reduce many 3-class outputs to one S/NS**.

### Files


| File                                                                        | Role                                        |
| --------------------------------------------------------------------------- | ------------------------------------------- |
| `data/dev_labeled_ChatGPT.jsonl`                                            | Same labelled facts as Part 1               |
| `data/passages_bm25_ChatGPT_humfacts.jsonl`                                 | Same BM25 Wikipedia chunks                  |
| `factchecking_main.py`                                                      | Loads DeBERTa; do not change                |
| `factcheck.py` → `EntailmentModel`, `EntailmentFactChecker`, `nli_premises` | Submitted Part 2                            |
| `Part2/tune_entailment.py`                                                  | First (failed) prune + threshold sweep      |
| `Part2/diagnose_entailment.py`                                              | Why aggressive prune was wrong              |
| `Part2/sweep_rules.py`                                                      | Decision-rule comparison on richer premises |
| `Part2/entailment_rules.json`                                               | Logged accuracies for those rules           |




### Model

`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

Label order used by this checkpoint (and by our code):

```text
0 = entailment
1 = neutral
2 = contradiction
```

We never fine-tune it. `model.eval()` + `torch.no_grad()` only.

---



## 2. Exploratory data analysis (EDA) for entailment

Part 1 EDA still applies (short facts, `<s>` markup, title-name traps). Extra observations that **changed Part 2**:

### 2.1 Passages are not sentences

A typical chunk is several `<s>…</s>` Wikipedia sentences glued together, plus headings such as `Career.` FEVER-style NLI expects a **sentence-like premise**. Running the model on the raw chunk was treated as a last-resort extra premise, not the main one.

Sentence counts after splitting (from `diagnose_entailment.py`):


|        | Sentences per fact |
| ------ | ------------------ |
| Min    | 13                 |
| Median | 26                 |
| Mean   | ~27.7              |
| Max    | 47                 |


So a naive “score every sentence” run is on the order of **6,000** premise–hypothesis pairs, not 221.

### 2.2 Pronouns and local context

Facts often use `He` / `She` / `They`. Wikipedia sentences after the lead may say `He founded the Fischer Research Laboratory` without the name. A model that never sees the **page title** can fail to resolve the pronoun.

**Implication:** prefix the Wikipedia **title** onto premises when it is not already in the first few characters.

### 2.3 Support can sit across two sentences

Dates, titles, and appositions sometimes split: sentence A names the person; sentence B has the claim. Single-sentence premises miss that.

**Implication:** also score **adjacent two-sentence windows**.

### 2.4 Lexical overlap is not a safe kill-switch

Part 1 leftover: after title stripping, **8 gold-S facts have overlap 0**, and **39 gold-NS facts have overlap 0**.

The brief suggests: *if overlap is very low, skip NLI*. That is true for many NS rows, but **false for those 8 supported paraphrases**. Discarding overlap `< 0.12` dropped **47 / 221** facts (**8 S + 39 NS**) and was the main reason the first entailment prototype sat at **79.2%**, *below* word overlap.

**Implication:** do **not** throw whole facts away on zero overlap. Use NLI precisely when bags of words fail.

### 2.5 Word overlap already gets 81%

The remaining errors are mostly paraphrase, wrong-year/wrong-film co-occurrence, and inference. NLI should **raise recall of S** without exploding false positives. The submitted system moved recall of `S` from **83.9% (Part 1)** to **87.5% (Part 2)** on the official driver.

---



## 3. Data cleaning (passage → premises)

We do not edit labels. Cleaning is **premise construction**.


| Step                                                            | What                                     | Why                                                                              |
| --------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| Split on `</?s>`                                                | BM25 Wikipedia sentence tags             | Those tags already mark sentence-like units                                      |
| Also split on `(?<=[.!?])\s+`                                   | Normal sentence boundaries               | Some chunks have no `<s>` tags                                                   |
| `html.unescape` + strip tags                                    | Markup is not English                    | Same as Part 1                                                                   |
| NFKC + collapse whitespace                                      | Weird Unicode / double spaces            | Stable strings for dedupe                                                        |
| Drop length < 8                                                 | Empty tags, `Career.` stubs              | Noise, not premises                                                              |
| Drop `references.` / `see also.` / `external links.` / `notes.` | Boilerplate headings                     | Never entail a biography fact                                                    |
| Dedupe (casefold)                                               | Repeated lead sentences across BM25 hits | Avoid scoring the same pair 5 times                                              |
| Title prefix `"{title}. {sentence}"`                            | Coreference                              | Hypothesis “He is an inventor” needs a name in the premise                       |
| Adjacent sentence pairs                                         | Two-sentence windows                     | Support split across a boundary                                                  |
| Truncated raw passage (first 1,000 chars, first two passages)   | Brief: “any part of any of the passages” | Coarse backup if splitting drops a span                                          |
| Cap at **48** premises                                          | Memory / time                            | Full unlimited set scored **85.5%** in the sweep; cap still **84.6%** officially |


Worked premise set (schematic):

```text
Title: Gerhard Fischer (inventor)

Sentence:  "Gerhard Fischer is an inventor and entrepreneur …"
Premises:  "Gerhard Fischer (inventor). Gerhard Fischer is an inventor and entrepreneur …"

Next:      "He founded the Fischer Research Laboratory in 1936."
Pair:      "<sent1> <sent2>" with the same title prefix

Hypothesis (fact): "Gerhard Fischer is an inventor."
```

---



## 4. Preprocessing inside the NLI model

Unlike Part 1, we do **not** stem or drop stopwords before DeBERTa. The checkpoint was trained on natural English.

What we *do* pass through the Hugging Face tokenizer:


| Choice                 | Value                            | Why                                            |
| ---------------------- | -------------------------------- | ---------------------------------------------- |
| Pair encoding          | `tokenizer(premise, hypothesis)` | Standard MNLI format                           |
| Truncation             | `max_length=512`                 | Model limit; sentences fit, raw chunks get cut |
| Padding                | True in a mini-batch             | Needed for batch inference                     |
| `eval()` + `no_grad()` | Always                           | No training; saves memory                      |


Logits → probabilities with softmax. We keep both:

- **argmax class** (discrete)
- **P(\text{entailment})** (threshold family)

---



## 5. Mapping three classes to S / NS



### 5.1 Failed first design (do not use)

`Part2/tune_entailment.py`:

- prune facts with Part 1 overlap `< 0.12`
- keep only sentences with overlap `> 0`
- cap 24 sentences
- discrete any-entail or sweep P(\text{entailment})


| Rule                                | Accuracy                           |
| ----------------------------------- | ---------------------------------- |
| Word overlap (Part 1)               | **81.0%**                          |
| Discrete any-entail (pruned)        | **79.2%**                          |
| Best P(\text{ent}) threshold (0.10) | **80.5%**                          |
| Official driver with that design    | **79.2%** (72 TP, **40 FN**, 6 FP) |


Too conservative: recall of `S` collapsed to **64%**. Ranking by overlap **removed the paraphrases NLI is for**.

### 5.2 Rules that were actually compared

`sweep_rules.py` scored **all** title-prefixed sentences, two-sentence windows, and truncated passages (no fact-level prune). Selected rows from `entailment_rules.json`:


| Rule                                      | Acc       | TP  | FN  | FP  | TN  | Keep?                                          |
| ----------------------------------------- | --------- | --- | --- | --- | --- | ---------------------------------------------- |
| Discrete: any premise argmax = entailment | **85.5%** | 101 | 11  | 21  | 88  | **Yes — submitted family**                     |
| Max P(\text{ent}) ≥ 0.5                   | **85.5%** | 101 | 11  | 21  | 88  | Same behaviour here                            |
| Max P(\text{ent}) ≥ 0.3                   | 85.1%     | 103 | 9   | 24  | 85  | Slightly more FP                               |
| Max P(\text{ent}) ≥ 0.2                   | 82.8%     | 104 | 8   | 30  | 79  | Under 83% wait — 82.8% is *below* 83%          |
| Max P(\text{ent}) ≥ 0.05                  | 80.1%     | 106 | 6   | 38  | 71  | Too loose                                      |
| `S` unless any contradiction              | **38.9%** | 27  | 85  | 50  | 59  | Unusable                                       |
| P(\text{ent})-P(\text{con}) ≥ 0.5         | 55.7%     | 25  | 87  | 11  | 98  | Unusable                                       |
| Part 1 overlap                            | 81.0%     | 94  | 18  | 24  | 85  | Baseline                                       |
| Discrete **or** overlap                   | 81.9%     | 108 | 4   | 36  | 73  | FP explosion                                   |
| Overlap unless contradiction (hybrid)     | **86.9%** | 107 | 5   | 24  | 85  | Higher, but **not** a pure entailment decision |


**Why we did not submit the 86.9% hybrid.** The brief’s Part 2 mark is for converting **entailment / neutral / contradiction** into S/NS. Mixing the Part 1 threshold back in as the main rule blurs the two parts and is harder to justify as “textual entailment”. Discrete any-entailment is exactly the sentence in the PDF: *entailed if entailed by any part of any of the passages*.

On this checkpoint, **discrete argmax-entailment** and **max P(\text{ent}) ≥ 0.5** produced the **same** 101/11/21/88 counts, because if the predicted class is entailment then P(\text{ent}) is usually ≥ 0.5 among three classes. The submitted code implements **discrete first**, with the 0.5 threshold only as a fallback if no premise is labelled entailment (in practice the fallback almost never fires).

### 5.3 Frozen decision

For premises p_1,\ldots,p_k and fact h:

```text
for each premise p:
    (P_ent, label) = DeBERTa(p, h)
    if label == "entailment":
        return S
    track max P_ent
return S if max P_ent >= 0.5 else NS
```

Neutral and contradiction both count as **not this premise**. Contradiction is **not** used as a hard veto over other sentences: one contradicting clause elsewhere on the page should not cancel a clearly supporting sentence. The “any contradiction → NS” rule’s 38.9% accuracy confirmed that.

---

### 6.1 What we refused to prune


| Idea                                | What happened                           | Verdict      |
| ----------------------------------- | --------------------------------------- | ------------ |
| Drop fact if overlap < 0.12         | 8 gold-S zeros discarded; 79% acc       | **Rejected** |
| Drop fact if overlap < 0.05         | Same 8 S / 39 NS (those zeros sit at 0) | **Rejected** |
| Score only windows with overlap > 0 | Drops paraphrase premises               | **Rejected** |


`_PRUNE_OVERLAP_THRESHOLD = 0.0` in code is a leftover of that experiment: it never triggers.

### 6.2 What we *do* prune (noise and volume)

This is the pruning the grid actually sees:

- empty / tiny fragments
- references / see-also headings
- duplicate premises
- **at most 48** premises per fact
- mini-batches of **4** pairs
- `del` tensors, `gc.collect()`, `torch.cuda.empty_cache()` after each batch (the starter already required this for autograding)

That is **sentence-level and computational pruning**, not “skip NLI when words do not match”. Efficiency improved without deleting the 8 zero-overlap supports.

### 6.3 Runtime


| Run                          | Hardware            | Time                                               |
| ---------------------------- | ------------------- | -------------------------------------------------- |
| First pruned tuner           | RTX 4060 Laptop GPU | ~152 s for 848 NLI calls (then failed the 83% bar) |
| Full rule sweep              | same GPU            | ~94 s for all facts, batch 8                       |
| Official `entailment --cuda` | same GPU            | **~7 min** (batch 4 + empty_cache every batch)     |


`--cuda` is what the brief recommends. CPU would work but is not practical.

---



## 7. End-to-end inference (what `predict` does)

For one fact and its passages:

1. Build `nli_premises(passages)` (split, clean, title, pairs, truncated chunks, dedupe, cap 48).
2. If the list is empty → `NS`.
3. `check_entailment_batch(premises, fact)` on GPU if the model has that method.
4. If **any** predicted label is `entailment` → `S`.
5. Else if max P(\text{entailment}) ≥ 0.5 → `S`.
6. Else `NS`.

No extra files, no fine-tuning, no change to `factchecking_main.py`.

---



## 8. Official evaluation

Command:

```text
python factchecking_main.py --mode entailment --cuda
```

(First run downloads the DeBERTa weights from Hugging Face; later runs use the cache.)


| Quantity       | Part 1 overlap        | Part 2 entailment     |
| -------------- | --------------------- | --------------------- |
| Accuracy       | 179 / 221 = **81.0%** | 187 / 221 = **84.6%** |
| Confusion      | 94 / 18 ; 24 / 85     | **98 / 14 ; 20 / 89** |
| Precision `S`  | 79.7%                 | **83.1%** (98/118)    |
| Recall `S`     | 83.9%                 | **87.5%** (98/112)    |
| F1 `S`         | 81.7%                 | **85.2%**             |
| Precision `NS` | 82.5%                 | **86.4%** (89/103)    |
| Recall `NS`    | 78.0%                 | **81.7%** (89/109)    |
| F1 `NS`        | 80.2%                 | **84.0%**             |


The 84.6% official number is slightly under the 85.5% sweep because the submitted code **caps at 48 premises**; the sweep did not. Both beat **83%**.

Remaining errors (seed for Part 3):

- **False positives (20):** a Wikipedia sentence *entails a close but wrong claim* (nearby year, related film, occupation that appears in a list). DeBERTa is locally eager.
- **False negatives (14):** support needs multi-hop or a span we truncated; coreference still fails; FEVER-style “not in the evidence” → neutral even when a human would say S.

---



## 9. Why these choices vs the Grade A grid


| Grid expectation        | What we did                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| 3-class → S/NS strategy | Discrete **any-entailment**; threshold 0.5 as equivalent fallback                                     |
| Compare alternatives    | Discrete vs P(\text{ent}) cuts vs contradiction veto vs hybrids; logged in `entailment_rules.json`    |
| Sentence splitting      | `<s>` tags + `.!?` boundaries                                                                         |
| Noise removal           | Markup, short stubs, references headings, duplicates                                                  |
| Word-overlap pruning    | Fact-level prune **rejected** after measuring 8 gold-S zeros; sentence noise + 48-premise cap instead |
| No OOM                  | Batch size 4, max length 512, delete tensors, empty CUDA cache                                        |
| Analysis of choices     | Failed 79% prune run, then 85.5% unpruned discrete run, then 84.6% official cap                       |
| ≥ 83%                   | **84.6%** on the official driver                                                                      |


---



## 10. Files to look at


| Path                                                                                                                      | What it is                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `factcheck.py` → `sentence_units`, `nli_premises`, `three_class_to_supported`, `EntailmentModel`, `EntailmentFactChecker` | Submitted Part 2                                                                                             |
| `factchecking_main.py`                                                                                                    | Model load + accuracy (unchanged)                                                                            |
| `test_factcheck.py`                                                                                                       | Mapping tests + fake model: any supporting sentence → `S`; unrelated → `NS`; zero overlap can still call NLI |
| `Part2/entailment_rules.json`                                                                                             | Full rule table                                                                                              |
| `Part1_Word_Overlap_Process.md`                                                                                           | Part 1 companion log                                                                                         |
| `CW.pdf`                                                                                                                  | Requirements and feedback grid                                                                               |


Unit check (no GPU):

```text
python -m unittest test_factcheck.py -v
```

---

