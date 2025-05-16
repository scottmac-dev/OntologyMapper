# Ontology Label Mapping Tool

Lightweight Python program that semantically maps labels from a **source ontology** to a **target schema** using the [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model. 

It outputs the results as a JSON file and a formatted text report for review.

## Features

- Uses a pre-trained model (`all-MiniLM-L6-v2`) for semantic similarity
- Maps source labels to the closest matching target labels
- Supports custom similarity threshold (default: 0.75)
- Generates a summary report with success rate and unknown mappings

## Requirements

```bash
pip install sentence-transformers torch
```

## Usage

```bash
python ontology_mapper.py <source_labels.json> <target_labels.json> <output_path.json> [threshold]
```

### Parameters:
- `<source_labels.json>` – Path to the source ontology labels (JSON array)
- `<target_labels.json>` – Path to the target schema labels (JSON array)
- `<output_path.json>` – Path to save the output results (as JSON)
- `[threshold]` (optional) – A float between 0 and 1 that sets the similarity threshold (default: `0.75`)

## Example

```bash
python ontology_mapper.py assets/source_ontology_labels.json assets/target_schema_ontology.json output/output.json 0.7
```

## Output

### JSON (e.g., `output/output.json`)
```json
{
  "house": {
    "mapping": "building",
    "score": 0.814
  },
  "school": {
    "mapping": "UNKNOWN",
    "closest target": "road",
    "score": 0.452
  }
}
```

### Report (e.g., `output/report.txt`)
```
===== START REPORT =====

TITLE: Ontology Mapping Report 16-05-2025 12:34:56

SUMMARY:
    - Source labels mapped: 10
    - Target labels: 8
    - Threshold for success: 0.75
    - Success percentage: 70.0%
    - Labels mapped: 7
    - Labels unknown: 3

RESULTS:
    == SUCCESS ==
    - Source: HOUSE -> Target: BUILDING || Score: 0.814
    ...

    == UNKNOWN ==
    - Source: SCHOOL -> Target: UNKNOWN || Closest target: ROAD Score: 0.452

===== END REPORT =====
```
