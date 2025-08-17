# Mini Report: Data Linkage Pipeline

## Project Title
**Linking Pre-operative Radiology Reports to PROMs for Hip Replacement Patients**

## Purpose
This project aims to determine whether pre-operative radiology reports and images can be used to predict patient-reported outcomes (PROMs) following hip and knee replacement surgery. This script focuses on the **data linkage phase** using R.

## Data Sources Used
- `SMR01` — Hospital admissions (used to identify surgical episodes)
- `OPERA` — Surgical operations data (to confirm hip procedures)
- `PROMs` — 6-month post-op outcome data (e.g., EQ-VAS scores)
- `Radiology reports` — Text-based radiologist findings before surgery

All datasets were mock/test data and no identifiable information was uploaded.

## Pipeline Overview (`linkage_pipeline.R`)
1. **Load all datasets** using `readr::read_csv()`
2. **Standardise CHI identifiers** (trim white space, convert to character)
3. **Remove duplicates** and rows with missing key fields
4. **Standardise fields** (e.g., SIDE field in OPERA)
5. **Convert dates safely** with `lubridate::ymd()`
6. **Merge OPERA with SMR01** to identify surgical episodes
7. **Link PROMs data** using CHI, filtered to 6-month outcomes only
8. **Link radiology reports** and restrict to pre-op only
9. **Export final dataset** with selected variables for modelling

   ### Pipeline Overview (linkage_pipeline.R)
1. Load all datasets …
2. Standardise CHI identifiers …
…
9. Export final dataset …

```mermaid

flowchart LR
   RR[Radiology Reports] --> CHI
   SMR[SMR01 Admissions] --> CHI
   OP[OPERA Operations] --> CHI
   PR[PROMs Outcomes] --> CHI
   CHI --> TL[Temporal Logic & Cleaning]
   TL --> CSV[Final Linked Dataset (De-identified)]


```
## Output
- `linked_clean_dataset.csv`  
  Final pre-processed, linked dataset (not uploaded to GitHub due to SHAIP data rules)

## Notes
- Data linkage was performed using **CHI** as the unique identifier
- Script designed for **secure environments** — no real data is uploaded or shared

## Author
Somaia Elsheikhi  
MSc Health Data Science – NLP Project  
University of Aberdeen
