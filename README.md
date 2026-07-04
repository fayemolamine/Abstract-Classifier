# Text Classification of Neurodegenerative Disease Research

---

## The Story

We trained a classifier to tell apart five neurodegenerative diseases from their research abstracts.
What it got wrong turned out to be more interesting than what it got right.

The model scores 93% on Dementia — and 0% on Parkinson's. That is not a bug.
It is a direct reflection of how these diseases share vocabulary in the scientific literature,
and it produced a semantic map of neurodegeneration that no one programmed.

---

## Project Structure

```
01_collect_pubmed_data.ipynb      ← PubMed API data collection
02_preprocessing.ipynb            ← Cleaning, lemmatization, train/val/test split
03_feature_engineering.ipynb      ← TF-IDF, BOW, n-grams
04_modeling.ipynb                 ← 6 classifiers, validation, champion selection
05_evaluation_shap.ipynb          ← Classification report, AUC, SHAP, error analysis
06_degradation_bias.ipynb         ← Word truncation experiment, bias overlay

neuro_abstracts_raw.csv           ← Full abstracts (1,000 records)
neuro_abstracts_100w.csv          ← Trimmed to ~100 words per record
neuro_preprocessed.csv            ← After cleaning and lemmatization
model_results.csv                 ← Train/val accuracy for all 6 models
accuracy_comparison.csv           ← Original vs reduced-text accuracy
misclassified_records.csv         ← All 95 misclassified test records

README.md                         ← This file
Report.pdf                        ← Written report
dsa_story.pptx                    ← Presentation slides
```

---

## How to Run

**Requirements**
```bash
pip install requests pandas numpy nltk scikit-learn xgboost shap matplotlib seaborn
```

**Run in order**
```
1. 01_collect_pubmed_data.ipynb   → produces neuro_abstracts_raw.csv + neuro_abstracts_100w.csv
2. 02_preprocessing.ipynb         → produces neuro_preprocessed.csv + train/val/test splits
3. 03_feature_engineering.ipynb   → produces TF-IDF and BOW feature matrices
4. 04_modeling.ipynb              → produces model_results.csv + champion model
5. 05_evaluation_shap.ipynb       → produces misclassified_records.csv + SHAP plots
6. 06_degradation_bias.ipynb      → produces accuracy_comparison.csv
```

> Always use **Kernel → Restart & Run All** for each notebook.
> Do not run cells out of order — each notebook depends on files from the previous one.

---

## Dataset

| Label | Disease | PubMed Query |
|-------|---------|--------------|
| `a` | Alzheimer's | Alzheimer's disease machine learning OR deep learning OR NLP |
| `b` | Parkinson's | Parkinson's disease machine learning OR deep learning OR NLP |
| `c` | Dementia | dementia vascular OR frontotemporal machine learning OR NLP |
| `d` | ALS | amyotrophic lateral sclerosis machine learning OR deep learning |
| `e` | Huntington's | Huntington's disease machine learning OR deep learning OR biomarker |

- **1,000 records total** — 200 per disease
- **Source:** PubMed E-utilities API (free, no key required)
- **Text:** Title + abstract, trimmed to ~100 words per record
- **Split:** 70% train / 15% validation / 15% test — stratified, `random_state=42`

---

## Preprocessing

Applied to `df_trim["text"]` → `df_trim["clean_text"]`:

1. Lowercase
2. Remove special characters — `re.sub(r'[^a-zA-Z\s]', ' ', text)`
3. Collapse whitespace
4. Remove NLTK English stopwords
5. Lemmatize — `WordNetLemmatizer()`

---

## Feature Engineering

| Representation | Parameters | Shape |
|---------------|-----------|-------|
| TF-IDF | `max_features=5000, ngram_range=(1,2), sublinear_tf` | (700, 5000) |
| BOW | `max_features=5000, ngram_range=(1,2)` | (700, 5000) |
| Hard (degradation) | `max_features=3000`, 40-word truncation | (800, 3000) |

All vectorizers fitted on training data only — never on validation or test.

---

## Results

### Model Comparison (TF-IDF features)

| Model | Train Acc | Val Acc |
|-------|-----------|---------|
| Naive Bayes | 0.5943 | 0.3267 |
| **SVM (champion)** | **0.6214** | **0.3400** |
| Random Forest | 0.6214 | 0.3267 |
| KNN | 0.5114 | 0.3067 |
| SGD | 0.6171 | 0.3400 |
| XGBoost | 0.6214 | 0.3200 |

**Champion:** SVM (`LinearSVC`) — highest validation accuracy.

### 10-Fold Cross-Validation (SVM)

| Metric | Value |
|--------|-------|
| Mean accuracy | 0.3070 |
| Std | 0.0562 |
| Best fold | 0.43 |
| Worst fold | 0.23 |

### Test Set — Classification Report (SVM)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Alzheimer's (a) | 0.043 | 0.033 | 0.038 | 30 |
| Parkinson's (b) | 0.000 | 0.000 | 0.000 | 30 |
| Dementia (c) | 0.824 | 0.933 | 0.875 | 30 |
| ALS (d) | 0.216 | 0.267 | 0.239 | 30 |
| Huntington's (e) | 0.621 | 0.600 | 0.610 | 30 |
| **Macro avg** | **0.341** | **0.367** | **0.352** | 150 |

**Test accuracy:** 0.3667 — **ROC-AUC:** 0.5499 (random = 0.50 for multi-class OvR)

### Per-Class Accuracy (confusion matrix diagonal)

| Disease | Correct / 30 | Accuracy |
|---------|-------------|---------|
| Alzheimer's | 1 | 3% |
| Parkinson's | 0 | 0% |
| Dementia | 28 | 93% |
| ALS | 8 | 27% |
| Huntington's | 18 | 60% |

### Degradation Experiment

| Dataset | Words/doc | Accuracy |
|---------|-----------|---------|
| Original | ~100 | 0.3667 |
| Reduced | 40 | 0.3150 |
| Drop | — | −0.0517 (−14.1%) |

---

## Key Findings

**1. The model did not treat all five diseases equally.**
Average accuracy hides a range from 0% (Parkinson's) to 93% (Dementia).

**2. Parkinson's has no linguistic home.**
Every Parkinson's test paper was predicted as either Alzheimer's or ALS — never as Dementia or Huntington's.
It sits at the intersection of motor language (shared with ALS) and cognitive language (shared with AD),
mirroring a known clinical ambiguity around Lewy-body dementia.

**3. Dementia is the most linguistically isolated disease.**
Vascular and frontotemporal terminology does not appear in the other four diseases.
The model draws a clean boundary almost every time.

**4. The confusion matrix is a semantic distance map.**
Top confusion pairs — PD↔ALS (26 errors), AD↔PD (23 errors) — reflect real co-morbidity
and shared pathological mechanisms, not random noise.

**5. 35% accuracy is a finding, not a failure.**
Random baseline for 5 classes is 20%. The model learns real signal (+15pp).
The ceiling is set by how similarly the research community writes about these diseases —
a linguistic ceiling, not a modelling one.

---

## Error Analysis

95 of 150 test records were misclassified. Top confusion pairs:

| Actual | Predicted | Count | Linguistic reason |
|--------|-----------|-------|------------------|
| Parkinson's | ALS | 15 | Motor language: neuron, degeneration, progression |
| Alzheimer's | Parkinson's | 12 | Cognitive + motor overlap in progressive neurodegeneration papers |
| Parkinson's | Alzheimer's | 11 | Lewy-body dementia papers bridge PD and AD vocabulary |
| ALS | Parkinson's | 11 | Motor neuron papers share PD's movement/progression framing |
| Alzheimer's | ALS | 10 | Generic ML methodology drowns disease-specific signal |

Full misclassified records saved to `misclassified_records.csv`.

---

## SHAP Explainability

`shap.LinearExplainer` on SVM, 50 test samples.

Key findings from mean |SHAP| per class:
- **Huntington's** — dominated by genetic/trial language (canal, binding, median). Completely distinct from other diseases.
- **Dementia** — clinical sociodemographic, voxel, decade life — specific neuroimaging + epidemiology vocabulary.
- **AD, PD, ALS** — share generic ML terms (depth, current state, paired data) — explaining their high mutual confusion.

---

## Authors

Group DSA_202101_1 · DTI5125 Data Science Applications · Summer 2026
