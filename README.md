# **Predicting Surgical Selection for Arthroplasty: A Comparative Analysis of NLP Models**

**Author:** Somaia Elsheikhi
**Course:** PU5926/PU5927 Professional Placement in Health Data Science (2024/25)

---

## **1. Project Summary**

This project investigates the use of Natural Language Processing (NLP) models to predict the selection of patients for arthroplasty (joint replacement surgery) based on pre-operative clinical radiology reports. The core objective is to evaluate the performance of the advanced GatorTron transformer model on this classification task using a rigorous cross-validation approach.

**A Note on Data Privacy:** The project workflow was developed in two stages. The initial data linkage was performed on a private, sensitive NHS dataset (SHARP project). To ensure reproducibility, the core NLP modeling pipeline was then developed and validated using the publicly available MIMIC-III dataset.

---

## **2. Project Workflow & Repository Structure**

The repository is structured to reflect the two main stages of the project.

### **Part A: Sensitive Data Processing (Demonstration Only)**
*   **Code:** `src/private_data_processing/linkage_pipeline.R`
*   **Description:** This R script was used for cleaning and linking the sensitive SHARP dataset. **This code is for demonstration only and cannot be run.**
*   **Report:** A summary of this process is available in `reports/mini_report.md`.

### **Part B: Reproducible NLP Modeling Pipeline (MIMIC-III)**
*   **Code:** The primary analysis is in the Jupyter Notebook: `notebooks/02_GatorTron_Model_Evaluation.ipynb`. A preceding notebook, `notebooks/01_Data_Preprocessing.ipynb`, should be run first to prepare the data.
*   **Description:** This is the main, reproducible part of the project. It uses the public MIMIC-III dataset to train and evaluate the GatorTron model.

---

## **3. How to Reproduce the Analysis**

To run the public-facing analysis (Part B), please follow these steps in order:

### **a. Clone Repository & Install Dependencies**
```bash
git clone https://github.com/somaia404/Predicting-Surgical-Selection-for-Arthroplasty.git
cd Predicting-Surgical-Selection-for-Arthroplasty
pip install -r requirements.txt
