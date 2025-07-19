# 🦴 Hip Replacement Data Linkage Pipeline (SHAIP NLP Strand)

This repository contains the R code for linking multiple datasets in preparation for NLP-based modelling of Patient Reported Outcome Measures (PROMs) following hip replacement surgery. It is part of the SHAIP/ARCHERY project, focusing on predicting post-operative outcomes using pre-operative radiology reports.

---

## 📁 Files

- `linkage_pipeline.R`: Core R script to clean, standardise, and link datasets.
- `mini_report.md`: Summary of the key linkage steps and their rationale.
- `linked_clean_dataset.csv`: Final clean dataset ready for analysis *(mock version or omitted if sensitive)*.
- `mock_*.csv`: Mock datasets used for testing the linkage process.

---

##  Workflow Overview

### 1. Data Loading
- Imports surgical (OPERA), admission (SMR01), radiology, and PROMs datasets.

### 2. Pre-processing
- Standardises CHI format.
- Removes rows with missing or duplicate CHI values.
- Ensures dates and text fields are cleaned.

### 3. Cohort Identification
- Filters relevant hip replacement records from OPERA using operation codes and side.

### 4. Dataset Linkage
- Links OPERA with SMR01 to enrich surgical context.
- Joins PROMs using CHI and filters to 6-month follow-up scores.
- Ensures radiology reports are pre-operative.

### 5. Final Output
- Assembles a clean, linked dataset with key variables: operation date, side, report text, and EQ-VAS score.

---

##  Purpose

The cleaned dataset will be used to develop NLP models (e.g., GatorTron, SciSpacy) to:
- Predict EQ-VAS score and MCID attainment
- Explore text-based features associated with outcomes
- Evaluate model interpretability and performance

---

##  Data Note

All datasets used in this example are mock or pseudonymised and for demonstration only. Please do **not upload identifiable patient data** to GitHub.

---

##  Data

⚠️ Due to data protection restrictions, patient datasets are not included in this repository.  
Expected filenames (not provided):
- `mock_opera_hip_operations.csv`
- `mock_smr01_linked.csv`
- `mock_proms_hip_data_200patients.csv`
- `mock_hip_radiology_reports.csv`

---

##  Requirements

- R 4.2+
- Packages:
  - `dplyr`
  - `readr`
  - `lubridate`

---

##  Credits

Developed as part of the SHAIP Work-Based Learning Placement at the University of Aberdeen.  
NLP strand led by **Somaia Elsheikhi**.
