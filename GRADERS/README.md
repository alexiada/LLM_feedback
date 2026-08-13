# LLM Grading Workflows for Reflective Assignments

This repository contains five Jupyter notebooks ("graders"), each implementing a different workflow for using a locally deployed Large Language Model (LLM) to mark and provide feedback on reflective assignments.

## What each grader does

- **GRADER1** — Zero-shot rubric grading. The LLM is given the assignment brief, the marking rubric, and the essay, and returns a mark and feedback directly. No worked examples are provided. This is the baseline workflow.

- **GRADER2** — Few-shot imitation grading. The LLM is given previously marked example essays (with their marks and feedback) but not the formal rubric, and is expected to infer the marking standard from the examples.

- **GRADER3** — Rubric plus examples. Combines GRADER1 and GRADER2: the LLM receives both the formal rubric and previously marked examples, so the examples calibrate the marking while the rubric constrains it to the official criteria.

- **GRADER4** — Multi-pass rubric decomposition. Built with LangGraph, this workflow splits the assessment into separate LLM calls, one per rubric criterion (presentation, reflection, breadth, link with practice), with a conditional branch for whether the essay uses a named reflective model. An integrity/resolution node then checks consistency between evidence, feedback, and scores before the final grade is aggregated.

- **GRADER5** — Retrieval-augmented pairwise calibration. For each new essay, the two most similar previously marked essays are retrieved using TF-IDF cosine similarity. The LLM first compares the new essay against these two neighbours to estimate mark deltas, converts these into a numeric prior, then grades the essay using the rubric, the retrieved examples, the pairwise comparison, and the prior.

All graders load a quantised (4-bit NF4) open-weight LLM locally via Hugging Face Transformers and return marks and feedback in JSON format.

## Before running

Each notebook loads a specific local model via a hardcoded `MODEL_DIR` path, e.g.:

```python
MODEL_DIR = "/path/to/project/models/qwen2.5-7b"
```

**You will need to change this to match the model you have installed locally.** Update `MODEL_DIR` (and, where present, alternative commented-out paths for other models) to point to wherever you have downloaded the model weights on your own system. Similarly, update `ESSAY_DIR` (and `EXAMPLES_DIR`, `OUTPUT_DIR` where present) to point to your own essay and output folders.

The notebooks were originally run on an institutional HPC cluster; all paths in this repository have been anonymised to generic placeholders (`/path/to/project/...`) and will not work as-is.

## Requirements

- Python 3.10+
- `transformers`, `torch`, `bitsandbytes` (for 4-bit quantisation)
- `langchain-core`
- `langgraph` (GRADER4 only)
- `python-docx`

## Citation

If you use or adapt these notebooks, please cite the accompanying paper (see LICENSE for details).
