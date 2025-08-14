# **Predicting Surgical Selection for Arthroplasty: A Comparative Analysis of NLP Models**

**Author:** Somaia Elsheikhi

**Course:** PU5926/PU5927 Professional Placement in Health Data Science (2024/25)

---

## **1. Project Summary**

This project investigates the use of Natural Language Processing (NLP) models to predict the selection of patients for arthroplasty (joint replacement surgery) based on pre-operative clinical radiology reports. The core objective is to evaluate the performance of advanced transformer models on this classification task using a rigorous cross-validation approach.

**A Note on Data Provenance:** The project workflow involved two distinct data stages. The initial data linkage was performed on a private, sensitive NHS dataset (SHARP project). To create a shareable and reproducible pipeline, a **synthetic dataset** was then generated that mimics the statistical properties and structure of the original clinical text. All model training and evaluation presented here were performed on this synthetic dataset.

---

## **2. Acknowledgements & Contributions**

### **Acknowledgements**
This project is based on a foundational code script generously provided by Dr. Luke Farrow. His original work served as the starting point for this analysis. We also thank the developers of PyTorch, Hugging Face, scikit-learn, and other open-source libraries that made this work possible.

### **My Contributions and Modifications**
This project builds upon the original work to create a more robust and comprehensive machine learning pipeline. The key modifications include:

*   **Synthetic Data Generation:** A Python script was developed to generate a realistic, synthetic dataset of clinical text and patient outcomes, enabling a fully reproducible analysis while protecting patient privacy.
*   **Enhanced Code Structure:** The code was refactored into distinct, logical sections for data generation, model setup, training, and evaluation.
*   **Improved Evaluation Workflow:** The script now calculates and reports overall average metrics and their 95% confidence intervals across all cross-validation folds.
*   **Comprehensive Visualization:** New plotting functions were added to generate an aggregated ROC curve, average confusion matrix, and a calibration plot.
*   **Increased Robustness:** The script now includes error handling for critical operations like file loading.

---

## **3. Project Workflow & Repository Structure**

The repository is structured to reflect the project's workflow.

### **Part A: Sensitive Data Processing (Demonstration Only)**
*   **Code:** `src/private_data_processing/linkage_pipeline.R`
*   **Description:** This R script was used for cleaning and linking the original sensitive SHARP dataset. **This code is for demonstration only and cannot be run.**

### **Part B: Reproducible NLP Modeling Pipeline**
*   **Data Generation:** The script `src/public_synthetic_data/01_generate_synthetic_data.py` creates the dataset used for modeling.
*   **Model Evaluation:** The Jupyter Notebook `notebooks/02_Model_Evaluation.ipynb` performs the main analysis on the synthetic data.

---

## **4. How to Reproduce the Analysis**

This project is fully reproducible. Please follow these steps in order:

### **a. Clone Repository & Install Dependencies**
```bash
git clone https://github.com/somaia404/Predicting-Surgical-Selection-for-Arthroplasty.git
cd Predicting-Surgical-Selection-for-Arthroplasty
pip install -r requirements.txt

