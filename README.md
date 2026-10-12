
# LLM Fact Checking

A fact-checking system that evaluates claims against retrieved Wikipedia passages using lexical similarity and textual entailment.

## Coursework Components

- **Part 1 — Word Overlap:** Implements and evaluates lexical similarity methods.
- **Part 2 — Textual Entailment:** Uses a pretrained entailment model to assess factual support.
- **Part 3 — Error Analysis:** Analyses model errors, identifies patterns, and discusses limitations.

## Setup

Requires Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

## Project Structure

- `factcheck.py` — Fact-checking implementations.
- `factchecking_main.py` — Supplied evaluation script.
- `requirements.txt` — Python dependencies.
- `data/` — Development data and retrieved Wikipedia passages.

## Status

Work in progress. Experimental results and implementation details will be documented as development continues.