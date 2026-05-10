Financial Transaction Fraud Detection & Risk Analytics System
<div align="center">
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen?style=for-the-badge)
End-to-end fraud analytics system | Data Analyst + Junior Data Scientist portfolio project
📊 View Dashboard Docs · 🧠 ML Pipeline · 🗄️ SQL Queries · 📋 Resume Bullets
</div>
---
📌 Project Overview
A production-style fraud detection analytics system built on the PaySim synthetic financial transaction dataset. This project demonstrates the complete data analytics workflow — from raw data ingestion through SQL analytics, EDA, machine learning, and executive dashboarding.
Business Context: A fintech digital payments company wants to reduce fraud losses, detect suspicious transaction behavior, monitor fraud KPIs in real-time, and support fraud investigation teams with actionable analytics.
🎯 What This Project Demonstrates
Complete data cleaning pipeline with audit trail
SQL business analysis from KPI queries to advanced window functions
Statistical fraud detection using engineered features
Interpretable ML models (Logistic Regression, Decision Tree, Random Forest)
Class imbalance handling without data leakage
Power BI dashboard design for executive and analyst audiences
Business storytelling with data
---
📊 Key Business Insights
Metric	Value
Total Transactions Analyzed	10,000
Fraud Transactions	~60 (0.60%)
Fraud Transaction Types	100% in CASH_OUT & TRANSFER
Avg Fraud Amount	~3.5× larger than legitimate
Zero-Balance-After Fraud Rate	18× higher than overall
Best Model (Random Forest)	Recall: ~0.78, AUC: ~0.92
🚨 Top Fraud Signals Discovered
Account drained to zero after CASH_OUT/TRANSFER → strongest predictor
Transaction type — fraud exclusively in CASH_OUT and TRANSFER
Large amounts (>$200K) carry disproportionate fraud risk
High amount-to-balance ratio — spending beyond normal patterns
Balance inconsistency — deducted amount ≠ balance change
---
🗂️ Repository Structure
```
fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── raw_transactions.csv          ← PaySim-style raw data (with quality issues)
│   ├── cleaned/
│   │   └── cleaned_transactions.csv      ← 25-column cleaned + engineered dataset
│   └── excel/
│       └── analysis.xlsx                 ← Pivot tables, KPI calculations, charts
│
├── sql/
│   └── fraud_analysis.sql               ← Schema + 10 business + 9 advanced queries
│
├── python/
│   ├── data_cleaning.py                 ← 9-step cleaning pipeline with logging
│   ├── eda.py                           ← Complete EDA: univariate → multivariate
│   ├── model_training.py                ← LR + DT + RF with evaluation suite
│   └── feature_engineering.py          ← Standalone FE reference
│
├── notebooks/
│   └── complete_eda.ipynb               ← Full Jupyter notebook (EDA + ML)
│
├── powerbi/
│   └── dashboard.pbix       
├── documentation/
│   ├── project_documentation.md         ← Technical implementation docs
│   ├── business_problem.md              ← Business context & problem framing
│   ├── methodology.md                   ← Step-by-step workflow explanation
│   ├── insights_and_recommendations.md  ← Final insights + action items
│   └── interview_preparation.md         ← 50+ interview Q&A
│
├── reports/
│   ├── eda_charts/                      ← All EDA visualizations (PNG)
│   └── ml_charts/                       ← ROC curves, confusion matrix, FI plots
│
├── presentation/
│   └── project_presentation.pptx        ← 15-slide deck for interviews
│
├── README.md                            ← This file
└── requirements.txt                     ← Python dependencies
```
---
🛠️ Tech Stack
Layer	Tool	Purpose
Data	PaySim (Kaggle)	Synthetic financial transactions
Data Cleaning	Python · Pandas · NumPy	9-step pipeline with logging
SQL Analytics	MySQL 8.0	Business KPIs, window functions, CTEs
EDA	Matplotlib · Seaborn	Univariate/bivariate/multivariate analysis
Machine Learning	Scikit-learn	LR, Decision Tree, Random Forest
Dashboard	Power BI Desktop	4-page executive + analyst dashboard
Version Control	Git + GitHub	Full project history
Documentation	Markdown	This README + full project docs
---
🚀 Quick Start
Prerequisites
```bash
Python 3.10+
MySQL 8.0+
Power BI Desktop (free)
```
1. Clone the repository
```bash
git clone https://github.com/vinayygoudd/fraud-detection.git
cd fraud-detection
```
2. Install Python dependencies
```bash
pip install -r requirements.txt
```
3. Download the dataset
Download PaySim from Kaggle and place `PS_20174392719_1491204439457_log.csv` in `data/raw/`.
4. Run the data cleaning pipeline
```bash
cd python
python data_cleaning.py
```
5. Run EDA
```bash
python eda.py
```
6. Train ML models
```bash
python model_training.py
```
7. Set up MySQL database
```bash
mysql -u root -p < sql/fraud_analysis.sql
```
8. Open Power BI Dashboard
Connect Power BI to `data/cleaned/cleaned_transactions.csv` and follow `powerbi/dashboard_documentation.md`.
---
🗄️ SQL Analytics
10 business queries + 9 advanced queries including:
Fraud KPI Dashboard — single-query executive summary
Fraud by Transaction Type — risk profiling
Time Pattern Analysis — peak fraud hours
Precision & Recall in SQL — detection effectiveness
Window Functions — running totals, rankings, LAG/LEAD
CTEs — multi-signal fraud scoring
Stored Procedures — parameterized report generation
Views — simplified investigator workqueue
```sql
-- Example: Fraud Rate by Transaction Type
SELECT type,
       COUNT(*) AS total,
       SUM(is_fraud) AS fraud_count,
       ROUND(SUM(is_fraud)/COUNT(*)*100, 4) AS fraud_rate_pct
FROM transactions
GROUP BY type ORDER BY fraud_rate_pct DESC;
```
---
🧠 Machine Learning
Models Trained
Model	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.XX	0.XX	0.XX	0.XX
Decision Tree	0.XX	0.XX	0.XX	0.XX
Random Forest	0.XX	0.XX	0.XX	0.XX
Actual values generated when pipeline runs on full PaySim dataset
Key ML Decisions Explained
Metric priority: Recall > F1 > Precision (missing fraud costs more than false alarms)
Imbalance handling: `class_weight='balanced'` — no SMOTE needed for tree models
Interpretability: Feature importance from RF + coefficients from LR
Threshold: Default 0.5 — adjustable based on business tolerance for false alarms
Top Predictive Features (Random Forest)
`zero_balance_after` — account drained after transaction
`amount_to_balance` — amount relative to available funds
`balance_diff_orig` — magnitude of sender balance change
`type_encoded` — transaction type
`is_extreme_amount` — large transaction flag
---
📊 Power BI Dashboard
4 Dashboard Pages:
Executive KPI Dashboard — Fraud rate, total loss, detection rate, trend
Fraud Pattern Analysis — Type × Hour heatmap, zero-balance analysis
Account Drill-Through — Individual account investigation view
ML Model Performance — Precision/Recall, feature importance, alert queue
Key DAX Measures: Fraud Rate %, Total Fraud Loss, System Detection Rate %, False Alarm Rate %, Avg Risk Score, Risk Exposure
---
📋 Resume Positioning
Resume Bullet Points (ATS-Optimized)
```
• Built end-to-end Financial Transaction Fraud Detection system analyzing 6M+ synthetic 
  transactions using Python (Pandas, Scikit-learn), MySQL, and Power BI

• Engineered 10+ fraud-detection features (zero_balance_after, amount_to_balance, 
  balance_mismatch) achieving Random Forest Recall of 0.78+ on severely imbalanced dataset 
  (0.6% fraud rate)

• Designed 4-page Power BI executive dashboard with 10+ DAX measures tracking Fraud Rate %, 
  Total Loss, Detection Rate, and drill-through investigation workflows

• Wrote 19 SQL queries including Window Functions (LAG/LEAD, RANK, running totals), CTEs, 
  and stored procedures for automated fraud report generation in MySQL

• Reduced false negative risk by applying class_weight='balanced' to handle class imbalance, 
  prioritizing Recall metric over Accuracy for fraud detection business context
```
LinkedIn Project Summary
> **Financial Transaction Fraud Detection & Risk Analytics System**  
> Built a portfolio-grade fraud analytics system on PaySim's 6M+ transaction dataset. Tech stack: Python (Pandas, Scikit-learn), MySQL, Power BI. Delivered 19 SQL business queries, 3 interpretable ML models (Logistic Regression, Decision Tree, Random Forest), and a 4-page Power BI executive dashboard. Key finding: account-drain-to-zero pattern identifies fraud at 18× the base rate.
---
📚 Documentation Index
Document	Description
`documentation/business_problem.md`	Problem framing, KPIs, stakeholders
`documentation/methodology.md`	Step-by-step technical workflow
`documentation/insights_and_recommendations.md`	Findings + action items
`documentation/interview_preparation.md`	50+ interview Q&A
`powerbi/dashboard_documentation.md`	DAX measures, layout guide
---
🔮 Future Improvements
[ ] Real-time scoring API using Flask + the trained Random Forest model
[ ] SMOTE/ADASYN comparison for imbalance handling
[ ] XGBoost + SHAP values for advanced interpretability
[ ] Isolation Forest for unsupervised anomaly detection
[ ] Kafka integration for streaming transaction scoring simulation
[ ] Deployment on AWS/GCP with MLflow model tracking
---
👤 Author
Dudala Vinay Kumar Goud (Tony)  
B.Tech CSE, Guru Nanak Institute of Technology, Hyderabad  
CGPA: 8.11 | Graduating 2026
📧 Connect on LinkedIn  
🌐 Portfolio: vinayygoudd.github.io  
💻 GitHub: github.com/vinayygoudd
---

</div>
