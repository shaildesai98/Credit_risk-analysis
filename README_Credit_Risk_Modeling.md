# Credit Risk Modeling: Predicting Loan Default

A machine learning project in R that predicts whether a borrower will default on a loan, using Logistic Regression and Random Forest classifiers evaluated on a real-world credit dataset.

---

## Business Problem

Financial institutions face significant losses from loan defaults. The goal of this project is to build and evaluate classification models that can accurately identify high-risk borrowers **before** a loan is issued — enabling better lending decisions and reducing credit risk exposure.

---

## Data Source

- **Dataset from Kaggle:** `credit_risk_dataset.csv`   
- **Rows:** 32,581 borrower records  
- **Features:** 12 columns covering borrower demographics, loan characteristics, and credit history

| Feature | Description |
| :---- | :---- |
| `person_age` | Borrower's age |
| `person_income` | Annual income |
| `person_home_ownership` | RENT / OWN / MORTGAGE |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose of loan (education, medical, etc.) |
| `loan_grade` | Credit grade assigned (A–G) |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate on loan |
| `loan_status` | **Target** — 0: No Default, 1: Default |
| `loan_percent_income` | Loan amount as % of income |
| `cb_person_default_on_file` | Prior default on credit bureau record |
| `cb_person_cred_hist_length` | Length of credit history (years) |

---

## Methodology

### 1\. Data Cleaning

- Imputed missing values in `person_emp_length` and `loan_int_rate` with their respective medians  
- Removed 165 duplicate rows  
- Removed biologically implausible outliers (`person_age > 100`, `person_emp_length > 60`)  
- Final clean dataset: **32,409 rows**

### 2\. Exploratory Data Analysis (EDA)

- Distribution plots for all numerical and categorical features  
- Boxplots segmented by loan default status  
- Bivariate scatter analysis (loan amount vs. interest rate, loan-to-income ratio)  
- Correlation heatmap of numerical features

### 3\. Preprocessing

- 80/20 train-test split (stratified by target)  
- Handled class imbalance (\~78% No Default / 22% Default) via **up-sampling** to achieve a balanced 50/50 training set

### 4\. Models Trained

- **Logistic Regression** — interpretable baseline using 5-fold cross-validation  
- **Random Forest** — ensemble model with hyperparameter tuning (`mtry` ∈ {3, 4, 5}, 300 trees), selected via ROC

### 5\. Threshold Tuning

- Applied **Youden's Index statistic** to find the optimal classification threshold (0.3217) for the best model, balancing sensitivity and specificity

---

## Results

| Metric | Logistic Regression | Random Forest |
| :---- | :---- | :---- |
| AUC | 0.8766 | **0.9367** |
| Accuracy | 0.8162 | **0.9296** |
| Sensitivity (Recall) | **0.7763** | 0.7382 |
| Specificity | 0.8274 | **0.9832** |
| Precision | 0.5572 | **0.9248** |
| F1 Score | 0.6488 | **0.8210** |

**Best Model: Random Forest** (AUC \= 0.9367)

After threshold tuning, the final model achieves **90.4% accuracy**, with sensitivity of 0.80 and specificity of 0.93 on the held-out test set.

### Key Findings

- **Random Forest** outperforms Logistic Regression across AUC, F1, and precision by capturing non-linear feature interactions.  
- **Logistic Regression** remains a competitive and interpretable baseline — valuable for regulatory contexts requiring model explainability.  
- The **most important predictors** of default are: loan interest rate, loan grade, loan-to-income ratio, and prior credit bureau default history.  
- Threshold tuning via Youden's J improves real-world default detection without excessive false positives.

---

## 💡 Business Solution: Who Should We Not Lend To?

The model answers this question directly. Based on the Random Forest analysis, **a borrower is high-risk and should be declined (or offered stricter terms) when they exhibit the following profile:**

### 🚩 High-Risk Borrower Profile

| Risk Factor | Danger Signal | Why It Matters |
| :---- | :---- | :---- |
| **Loan Grade** | Grade D, E, F, or G | Grades assigned by lenders already encode creditworthiness — lower grades correlate strongly with default |
| **Interest Rate** | Above \~13–15% | High rates reflect high perceived risk; borrowers already flagged as risky by the system |
| **Loan-to-Income Ratio** | Above 40–50% of income | Borrower cannot realistically repay without financial strain |
| **Prior Default on File** | Yes (`cb_person_default_on_file = Y`) | Historical default behavior is the single strongest behavioral signal |
| **Loan Purpose** | Medical or personal (debt consolidation) | These purposes correlate with financial distress, not investment |
| **Home Ownership** | Renter with low income | No asset collateral; higher vulnerability to income shocks |
| **Employment Length** | Very short (\< 1 year) | Income instability increases repayment risk |

### ✅ What the Model Recommends

- **Decline** borrowers who trigger 3 or more high-risk signals above, especially if loan grade is D–G **and** loan-to-income ratio exceeds 50%.  
- **Approve with caution** (lower limits, higher collateral) borrowers with 1–2 risk flags and a clean credit bureau history.  
- **Approve confidently** borrowers with Grade A–B loans, loan-to-income ratio under 20%, no prior default, and stable employment.

