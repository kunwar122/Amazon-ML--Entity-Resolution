# Business Entity Resolution

**Amazon ML Challenge 2026** | **F0.5 score: 0.86**

A machine learning pipeline that matches noisy business records across three independent data sources, without any shared identifier. For each business in a reference source (Source 1), the model finds all records for the same real-world business in Source 2 and Source 3.

---

## Problem

Each source stores a business name, address and country, but the same business can look very different between sources: abbreviations, reordered or missing address parts, legal suffixes (Inc, Ltd, Pvt), and names written in different scripts. A Source 1 business can have zero, one or several matches.

The score is a **macro F0.5** computed per Source 1 entity, so precision counts twice as much as recall. Wrong merges are penalized more than missed matches, and correctly predicting "no match" scores full marks.

## Results

| Metric | Value |
| --- | --- |
| F0.5 score | **0.86** |
| Model | LightGBM (gradient-boosted trees) |
| Features per candidate pair | 16 |

## Data

| | Rows |
| --- | --- |
| Test: Source 1 (reference) records | 1,732,544 |
| Test: Source 2 + Source 3 candidate records | 9,969,589 |
| Train: Source 1 records | 2,206,821 |
| Train: Source 2 + Source 3 records | 10,320,219 |

- Countries: US and India in training, with France added in the test set.
- Files are tab-separated (TSV) and include `business_name`, `business_address` and `country`.
- The dataset is **not included** in this repository. Download it from the challenge page.

## Approach

1. **Data preparation.** Names are lowercased, accent-stripped and cleaned of punctuation, and legal suffixes are removed to get a core name. Addresses are normalized (for example `rd` to `road`), and a 5-6 digit postal code is extracted.
2. **Candidate generation.** Comparing every pair of records is not feasible at this scale, so a shortlist is built for each record:
   - character n-gram TF-IDF similarity on names within each country (top 15 per record)
   - exact country and postal code matches
   - a relaxed cross-country search for records that got no candidates
3. **Feature engineering.** 16 similarity features are computed for each candidate pair (see below).
4. **Modeling.** A LightGBM classifier is trained on the pair features. The train/validation split is grouped by Source 1 entity, so no entity appears in both sets.
5. **Threshold tuning.** The decision threshold is chosen by sweeping it against the competition's F0.5 metric directly, not against accuracy or AUC.
6. **Inference and checks.** Predictions are written as TSV submission files and checked for format, duplicates and valid IDs.

## Features (16)

| Group | Count | Features |
| --- | --- | --- |
| Name | 10 | edit-distance ratio, token sort ratio, token set ratio, partial ratio, token Jaccard, character bigram Jaccard, full-name ratio (suffixes kept), length difference, common prefix length, first-token match |
| Address | 3 | token sort ratio, token Jaccard, character bigram Jaccard |
| Structured | 3 | postal code match, country match, TF-IDF cosine score from candidate generation |

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn (TF-IDF), sparse_dot_topn, RapidFuzz, LightGBM

## Repository structure

```
.
├── entity_resolution_pipeline.ipynb   # full pipeline: data prep to submission
├── requirements.txt
└── README.md
```

## How to run

1. Download the challenge data and note where it is stored.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `entity_resolution_pipeline.ipynb` and update `BASE_TRAIN`, `BASE_TEST` and `OUTPUT_DIR` in the configuration cell.
4. Run all cells. The notebook writes `matching_results.tsv` and `candidate_pairs.tsv` to the output folder.

The notebook was developed and run on Kaggle.

## Limitations and future work

- Matches are limited to the shortlist from candidate generation, so any true match missing from it cannot be recovered. Raising `TOPN` or lowering `LOWER_BOUND` would increase coverage.
- Short, generic business names with little address overlap are a likely source of false positives. A targeted feature for this case could help.
- Possible improvements: tuning `scale_pos_weight` for higher precision, and ensembling LightGBM with CatBoost or XGBoost.

## Team

- Kunwar Maharana Pratap Singh, [github.com/kunwar122](https://github.com/kunwar122)
- Milind Kumar Mandal, [github.com/milind2109](https://github.com/milind2109)
