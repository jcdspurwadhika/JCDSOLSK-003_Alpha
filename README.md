# E-Commerce Customer Churn Prediction  
JCDSOLSK-003_Alpha

## Project Overview
Customer churn is one of the most critical challenges in e-commerce businesses, as acquiring new customers is generally more expensive than retaining existing ones.  

This project aims to build a machine learning–based classification system to predict whether a customer is likely to churn, and to translate model outputs into actionable business insights that can reduce retention and acquisition costs.

The project covers:
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Model benchmarking and evaluation
- Cross-validation and hyperparameter tuning
- Imbalanced data handling
- Model interpretability using SHAP
- Business impact simulation
- Deployment using Streamlit
- Visualization using Tableau

---

## Business Objective
- Identify customers with high churn risk.
- Optimize retention campaign targeting.
- Reduce unnecessary retention cost.
- Minimize revenue loss from undetected churn customers.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution of churn vs non-churn customers  
- Analysis of behavioral features (tenure, complaints, satisfaction, order activity, etc.)  
- Identification of important patterns related to churn

### 2. Feature Engineering & Preprocessing
- One-Hot Encoding for categorical features  
- Separate pipelines:
  - With scaling → Logistic Regression, KNN  
  - Without scaling → Decision Tree, Random Forest, XGBoost  
- Train-test split: 80% training, 20% testing (stratified)

### 3. Model Benchmarking
Models evaluated:
- Logistic Regression  
- KNN  
- Decision Tree  
- Random Forest  
- XGBoost  

Primary evaluation metric:
- **F1-score** (balances False Positive and False Negative)

### 4. Cross-Validation
- 5-fold cross-validation  
- Mean and standard deviation of F1-score reported

### 5. Hyperparameter Tuning
- GridSearchCV applied to all models  
- Best parameters selected based on cross-validated F1-score

### 6. Imbalanced Data Handling
Techniques compared:
- No Sampling  
- NearMiss (undersampling)  
- SMOTE (oversampling)  
- SMOTE-ENN (hybrid)

Best approach selected based on cross-validated F1-score.

### 7. Final Model
- XGBoost selected as final model  
- High recall and precision for churn class  
- Strong overall generalization

### 8. Model Interpretability
- SHAP (SHapley Additive Explanations) used  
- Top drivers of churn:
  - Tenure  
  - Complaint  
  - Number of Addresses  
  - Cashback Amount  
  - Warehouse to Home distance  
  - Days Since Last Order  
  - Satisfaction Score  

---

## Final Model Performance (Test Set)

- Recall (Churn): 96%  
- Precision (Churn): 99%  
- Recall (Non-Churn): 100%  
- Accuracy: 99%

Confusion Matrix:

|            | Pred Non-Churn | Pred Churn |
|------------|---------------|------------|
| Actual Non-Churn | 934 | 2 |
| Actual Churn | 8 | 182 |

---

## Business Impact Simulation

Assumptions:
- Total customers: 1,126  
- Churn customers: 190  
- Non-churn customers: 936  
- Retention cost per customer: IDR 500,000  
- Acquisition cost per lost customer: IDR 500,000  

### Without Machine Learning
- All customers receive retention campaign  
- Total cost:  
  1,126 × 500,000 = IDR 563,000,000  

### With Machine Learning
- Retention cost applied only to TP and FP  
- Acquisition cost applied to FN  

Retention Cost:
- (TP + FP) × 500,000  
- (182 + 2) × 500,000 = IDR 92,000,000  

Acquisition Cost:
- FN × 500,000  
- 8 × 500,000 = IDR 4,000,000  

Total Cost with ML:
- IDR 96,000,000  

### Cost Saving
- IDR 563,000,000 – IDR 96,000,000  
- **IDR 467,000,000 (~83% reduction)**

---

## Deployment

### Streamlit App
Live demo:  
https://alpha-ml-churn.streamlit.app/

Features:
- Single customer prediction  
- Batch prediction via CSV upload  
- Probability output and churn classification  

### Tableau Dashboard
- Churn rate KPI  
- Total customers  
- Total churn customers  
- Churn by tenure  
- Churn by satisfaction score  
- Churn by complaint  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn  
- SHAP  
- Streamlit  
- Tableau  

---

## Team Alpha
JCDSOLSK-003 – Purwadhika Digital Technology School
Ardhian Dewagupta Pratama
Cakraningrat Kencana Murti
Priadi Jatmiko

---

## Key Takeaways
- Machine learning significantly improves retention targeting efficiency  
- Combining predictive modeling with EDA-driven insights enables both **who to target** and **how to retain**  
- The system functions as a **decision support tool**, not a fully automated decision maker
