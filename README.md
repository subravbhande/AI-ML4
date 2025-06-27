# 🚦 Traffic Accident Fatality Classification using Logistic Regression

This project demonstrates how to build a **binary classification model** using **Logistic Regression** on real-world traffic accident data. The model predicts whether a region has **high fatality rates** based on accident statistics using `scikit-learn`, `pandas`, and `matplotlib`.

---

## 📂 Dataset

- **File Name**: `ADSI_Table_1A.2.csv`  
- **Source**: [Accidental Deaths & Suicides in India - NCRB](https://ncrb.gov.in)  
- **Columns**: Traffic-related injury/death counts by State/UT

---

## 🎯 Objective

To classify regions as:
- `1` (**High Fatality**) if `Total Traffic Accidents - Died > 5000`
- `0` (**Low Fatality**) otherwise

---

## 🧰 Tools & Libraries

- Python 3
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## 📊 Features Used

Features include accident statistics such as:
- `Road Accidents - Cases`, `Injured`, `Died`
- `Railway Accidents - Cases`, `Injured`, `Died`
- `Railway Crossing Accidents - Cases`, `Injured`, `Died`
- `Total Traffic Accidents - Cases`, `Injured`, `Died`

---

## 🔁 Workflow

1. **Data Loading & Preprocessing**
   - Load dataset using `pandas`
   - Drop irrelevant columns (`Sl. No.`, `State/UT/City`)
   - Create binary target column `HighFatality`

2. **Train-Test Split & Scaling**
   - 80-20 data split using `train_test_split`
   - Feature standardization using `StandardScaler`

3. **Model Training**
   - Fit a `LogisticRegression` model

4. **Model Evaluation**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - ROC Curve & AUC Score
   - Threshold tuning (e.g., `0.3`)
   - Feature importance from model coefficients

---

## 📈 Model Performance

Example Metrics:

Confusion Matrix:
[[2 0]
[1 2]]

Classification Report:
precision recall f1-score support
0 0.67 1.00 0.80 2
1 1.00 0.67 0.80 3

ROC-AUC Score: 0.83



---

## 📌 Key Concepts

- **Logistic Regression**: Linear model for binary classification
- **Sigmoid Function**: Converts output to probability between 0 and 1
- **Threshold Tuning**: Adjust decision boundary (default is 0.5)
- **ROC Curve**: Plots true positive rate vs. false positive rate
- **AUC Score**: Area under ROC curve — higher is better

---

## 📁 Files in Repository

| File Name                     | Description                                |
|------------------------------|--------------------------------------------|
| `logistic_classification.py` | Python script with full implementation     |
| `ADSI_Table_1A.2.csv`         | Input dataset file (if allowed to upload)  |
| `README.md`                  | Project documentation                      |
| `confusion_matrix.png`       | Optional — visualization of confusion matrix |

---

## ✅ How to Run the Project

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python logistic_classification.py
