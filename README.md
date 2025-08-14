# **Predicting Surgical Selection for Arthroplasty: A Comparative Analysis of NLP Models**

**Author:** Somaia Elsheikhi

**Course:** PU5926/PU5927 Professional Placement in Health Data Science (2024/25)

---

## **1. Project Summary**

This project investigates the use of Natural Language Processing (NLP) models to predict the selection of patients for arthroplasty (joint replacement surgery) based on pre-operative clinical radiology reports. The core objective is to evaluate the performance of advanced transformer models on this classification task using a rigorous cross-validation approach.

**A Note on Data Privacy:** The project workflow was developed in two stages. The initial data linkage was performed on a private, sensitive NHS dataset (SHARP project). To ensure reproducibility, the core NLP modeling pipeline was then developed and validated using the publicly available MIMIC-III dataset.

---

## **2. Acknowledgements & Contributions**

### **Acknowledgements**
This project is based on a foundational code script generously provided by Dr. Luke Farrow. His original work served as the starting point for this analysis.

We would also like to thank the developers of the following essential libraries and resources:
*   **PyTorch & Transformers:** For the powerful deep learning framework and access to pre-trained models.
*   **scikit-learn:** For the comprehensive suite of machine learning tools used for evaluation.
*   **Hugging Face:** For their contributions to the NLP community, particularly the GatorTron model and the Hugging Face Hub.
*   **matplotlib & seaborn:** For their invaluable data visualization capabilities.

### **My Contributions and Modifications**
This project builds upon the original work to create a more robust and comprehensive machine learning pipeline. The key modifications include:

*   **Enhanced Code Structure:** The code was refactored into distinct, logical sections for data loading, model setup, training, and evaluation, improving readability and maintainability.
*   **Improved Evaluation Workflow:** The script now calculates and reports overall average metrics and their 95% confidence intervals across all cross-validation folds, providing a more reliable and statistically sound assessment of model performance.
*   **Comprehensive Visualization:** New plotting functions were added to generate an aggregated ROC curve, average confusion matrix, and a calibration plot based on predictions from all folds, offering a more complete view of the model's behavior.
*   **Increased Robustness:** The script now includes error handling for critical operations like file loading, ensuring it fails gracefully with informative messages.
*   **External Model Loading:** The model is now loaded directly from the Hugging Face Hub, making the script more portable and easier for others to reproduce.

---

## **3. Project Workflow & Repository Structure**

The repository is structured to reflect the two main stages of the project.

### **Part A: Sensitive Data Processing (Demonstration Only)**
*   **Code:** `src/private_data_processing/linkage_pipeline.R`
*   **Description:** This R script was used for cleaning and linking the sensitive SHARP dataset. **This code is for demonstration only and cannot be run.**
*   **Report:** A summary of this process is available in `reports/mini_report.md`.

### **Part B: Reproducible NLP Modeling Pipeline (MIMIC-III)**
*   **Code:** The primary analysis is in the Jupyter Notebook: `notebooks/02_GatorTron_Model_Evaluation.ipynb`. A preceding notebook, `notebooks/01_Data_Preprocessing.ipynb`, should be run first to prepare the data.
*   **Description:** This is the main, reproducible part of the project. It uses the public MIMIC-III dataset to train and evaluate the GatorTron model.

---

## **4. How to Reproduce the Analysis**

To run the public-facing analysis (Part B), please follow these steps in order:

### **a. Clone Repository & Install Dependencies**
```bash
git clone https://github.com/somaia404/Predicting-Surgical-Selection-for-Arthroplasty.git
cd Predicting-Surgical-Selection-for-Arthroplasty
pip install -r requirements.txt
