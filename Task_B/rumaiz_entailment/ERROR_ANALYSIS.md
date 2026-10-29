# Part 3: Error Analysis of the Entailment Model (draft)

**Model analysed:** DeBERTa-v3 (MNLI + FEVER + ANLI), 2-sentence window premises, fact score = max P(entailment), fact-level word-overlap pruning (τ = 0.1), threshold θ = 0.314 → **86.4% accuracy** (191/221).

**Errors made:** 30 in total, 12 false positives (FP) and 18 false negatives (FN). Source: `entailment_errors.json`.

**Sample examined:** 10 FP and 10 FN, as the brief requires. Two duplicates and two likely annotation errors were set aside (see §5) to keep the sample representative.

---

## 1. Error categories

| # | Category | Definition |
|---|---|---|
| **A** | **Topical overlap without the claimed relation** | The evidence mentions the same entities or topic as the fact, but asserts a *different* relation (e.g. *studied at* vs *worked at*, *Under-20* vs *Under-19*). The model treats strong topical similarity as entailment. |
| **B** | **Vague, generic or world-knowledge claims** | The fact is too general, evaluative or common-sense to be stated explicitly in the passage (*"worked with a number of organisations"*, *"Cleveland is in Ohio"*, *"X is popular"*). The model either over-generalises from loosely compatible evidence (FP) or finds nothing explicit enough (FN). |
| **C** | **Evidence in non-sentential or noisy text** | The supporting information exists, but in a **list, bibliography, honours table or broken sentence** rather than a well-formed sentence, so no premise window states the fact in a form the NLI model can match. |
| **D** | **Surface-form mismatch, amplified by pruning** | The fact and the evidence refer to the same thing with **different surface forms** (spelling variant *Fischer/Fisher*, *Argentinian/Argentine*, *United Kingdom/London, England*). Low word overlap causes the pruning step to reject the fact before the model sees it, or the model fails to align the forms. |

---

## 2. Aggregate statistics

| Category | False positives | False negatives | Total |
|---|---|---|---|
| A: Topical overlap without the claimed relation | 4 | 0 | 4 |
| B: Vague / generic / world-knowledge claims | 6 | 2 | 8 |
| C: Evidence in non-sentential or noisy text | 0 | 4 | 4 |
| D: Surface-form mismatch, amplified by pruning | 0 | 4 | 4 |
| **Total** | **10** | **10** | **20** |

**Key trends**

- **False positives are all about over-trusting similarity** (categories A and B). 9 of the 10 FPs have P(entailment) > 0.70, so the model is confidently wrong when the evidence is *about the right thing but says something different*.
- **False negatives are mostly about how the evidence is written**, not what it says (categories C and D: 8 of 10). The facts are true, but the evidence is in a list, uses a different spelling, or is filtered out by pruning.
- **Pruning cost:** 4 of all 18 FNs were pruned. In 2 of them (*Hibo Wardere … United Kingdom*, *Florencia Bertotti … Argentinian*), the model itself would have entailed the fact (P(E) = 0.999 and 0.993). Pruning still helps overall: it raises accuracy from 84.2% to 86.4% and skips 17% of model calls. But its errors are concentrated on surface-form mismatches.

---

## 3. Errors examined

### False positives (predicted S, gold NS / IR)

| # | Fact | Gold | P(E) | Category |
|---|---|---|---|---|
| FP1 | Cleveland is in Ohio. | IR | 0.993 | B |
| FP2 | She transitioned to acting. | NS | 0.812 | B |
| FP3 | Itakura has represented Japan at the Under-19-level. | NS | 0.513 | A |
| FP4 | Trivisonno worked at WMMS. | NS | 0.965 | A |
| FP5 | Trivisonno worked at WKNR. | NS | 0.993 | A |
| FP6 | She has worked with a number of organizations. | NS | 0.924 | B |
| FP7 | He has worked as a researcher at various organizations. | NS | 0.938 | B |
| FP8 | Mayo Clinic is an organization. | NS | 0.888 | B |
| FP9 | He has worked at Mayo Clinic. | NS | 0.709 | A |
| FP10 | Jidenna's music is a fusion of hip-hop. | NS | 0.857 | B |

### False negatives (predicted NS, gold S)

| # | Fact | P(E) | Pruned? | Category |
|---|---|---|---|---|
| FN1 | Gerhard Fischer is best known for inventing a metal detector. | 0.011 | no | D |
| FN2 | Gerhard Fischer invented the first handheld, battery-operated metal detector. | 0.005 | no | D |
| FN3 | Jean Daullé was known for his portraits. | 0.076 | no | B |
| FN4 | The Taker trilogy is a novel by Katsu. | 0.050 | no | C |
| FN5 | The Hunger is a novel by Katsu. | 0.006 | no | C |
| FN6 | He served for many years as a pastor. | 0.024 | no | C |
| FN7 | Hibo Wardere is based in the United Kingdom. | (0.999)* | **yes** | D |
| FN8 | With CSKA, he won domestic titles. | 0.035 | no | C |
| FN9 | Florencia Bertotti is an Argentinian. | (0.993)* | **yes** | D |
| FN10 | "New Heart." is popular. | 0.217 | no | B |

\* The probability the model *would* have given; the fact was pruned before the model was run.

---

## 4. Detailed examples

### Example 1: Topical overlap without the claimed relation (FP)
- **(a) Fact:** *"Trivisonno worked at WKNR."*
- **Best evidence:** *"…in the ratings, even though it was usually up against Cleveland Indians games on WKNR. It did so well, the decision was made to put Trivisonno in afternoon drive-time…"*
- **(b) Gold label:** NS  **(c) Predicted:** S (P(E) = 0.993)
- **(d) Category:** A, topical overlap without the claimed relation
- **(e) Why:** The passage mentions both Trivisonno and WKNR in a radio-career context, but WKNR is a *competing* station whose broadcasts his show was up against, not his employer. The model rewards the co-occurrence of the right entities in the right domain instead of checking the *worked at* relation between them.

### Example 2: Surface-form mismatch, amplified by pruning (FN)
- **(a) Fact:** *"Florencia Bertotti is an Argentinian."*
- **Evidence in the passage:** *"María Florencia Bertotti (born 15 March 1983)… is an **Argentine** actress, singer, songwriter and businesswoman."*
- **(b) Gold label:** S  **(c) Predicted:** NS (pruned, never sent to the model)
- **(d) Category:** D, surface-form mismatch amplified by pruning
- **(e) Why:** Once the subject's name is removed, the only content word left is *argentinian*, which is stemmed differently from *argentine* and so shares no token with the evidence. The word-overlap score falls below τ and the fact is pruned. Had it reached the model, DeBERTa would have entailed it with P(E) = 0.993, so this error comes entirely from the lexical pruning step.

### Example 3: Evidence in non-sentential text (FN)
- **(a) Fact:** *"The Hunger is a novel by Katsu."*
- **Evidence in the passage:** *"Bibliography.:Standalone novels. - "The Hunger", Transworld, March 2018 () - "The Deep", Transworld, March 2020 ()…"*
- **(b) Gold label:** S  **(c) Predicted:** NS (best P(E) = 0.006)
- **(d) Category:** C, evidence in non-sentential text
- **(e) Why:** The support is a bibliography list with no verb or author mention in the same unit. The fact that it's a novel *by Katsu* is implied only by the section header on Katsu's own page. NLI models are trained on sentence pairs, so a premise like *"- "The Hunger", Transworld, March 2018 ()"* gives them no proposition to entail.

---

## 5. Further observations and critical reflection

- **Possible annotation noise.** Two errors not included in the sample above suggest the model may be right and the gold label wrong:
  - *"He served as a sniper."* (gold NS) is supported verbatim: *"Rob Furlong … is a Canadian former military sniper."*
  - *"She was born on July 18, 1987."* (gold S) is contradicted by the passage: *"born in Maracaibo, Venezuela on 14 August 1987."*

  FActScore annotators judged facts against Wikipedia at annotation time, which may differ from the retrieved passages. This caps achievable accuracy and means some reported errors are not model failures.
- **The model is confident when wrong.** Most FPs have P(E) > 0.85, so raising θ would not fix them without losing many true positives. The errors come from *what* the model attends to (topical overlap), not from calibration.
- **Pruning trades recall for precision.** Lexical pruning removes many unsupported facts cheaply, but it inherits the word-overlap model's blindness to paraphrase and spelling variation (category D).

**Possible improvements**

1. **Normalise surface forms before pruning:** lemmatise demonyms (*Argentinian → Argentina*), use fuzzy name matching (*Fischer/Fisher*), or map place names to their country. Alternatively, prune only on a very low threshold.
2. **Convert lists and tables into sentences** before NLI, e.g. *"Katsu's standalone novels include The Hunger (2018)"*. Section headers could be prepended to list items.
3. **Relation-aware verification for category A:** require the fact's predicate to be supported (e.g. check that the subject–relation–object triple appears) rather than relying on entity co-occurrence. Using a larger NLI model or adding contradiction-aware scoring are cheaper options.
4. **Abstain on vague claims (category B):** facts like *"X is popular"* or *"worked with various organisations"* could be flagged as unverifiable rather than forced into S/NS.
