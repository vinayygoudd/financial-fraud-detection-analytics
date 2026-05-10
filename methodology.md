# Financial Transaction Fraud Detection
## Project Methodology & Technical Documentation

**Author:** Dudala Vinay Kumar Goud (Tony) | GNIT Hyderabad  
**Dataset:** PaySim Synthetic Financial Transactions (Kaggle)  
**Project Type:** End-to-end Data Analytics + Junior Data Science Portfolio Project

---

## Phase 1: Business Understanding

**Problem:** A fintech company's rule-based fraud flag system misses 78%+ of actual fraud transactions, leading to significant financial losses and poor customer trust.

**Success Criteria:**
- Identify top fraud indicators from transaction data
- Build a risk scoring system (0–100) to prioritize investigation
- Train ML models with Recall ≥ 0.75 and ROC-AUC ≥ 0.85
- Deliver executive dashboard with Fraud Rate %, Loss, Detection Rate KPIs

**Stakeholders:**
- Risk Management Leadership → Fraud Rate % and Loss KPIs
- Fraud Investigation Team → Alert queue and account drill-through
- Data Science Team → Model metrics and feature importance
- Operations → False alarm rate (investigator workload)

---

## Phase 2: Data Understanding

**Dataset:** PaySim simulates 6.3M mobile money transactions over 30 days.

**Key domain rules discovered:**
1. Fraud ONLY occurs in CASH_OUT and TRANSFER transactions
2. Fraudsters drain sender accounts to zero (new_balance_orig = 0)
3. The existing isFlaggedFraud column only catches large fraud (>$200K threshold)
4. Destination accounts of fraud transactions are often newly created (0 prior balance)

**Data quality issues found:**
- 120 missing values in oldbalanceDest
- 80 missing values in newbalanceDest
- 5 exact duplicate rows
- Right-skewed amount distribution (log-normal)
- Severe class imbalance: 0.60% fraud rate

---

## Phase 3: Data Preparation

**Cleaning decisions and rationale:**

| Issue | Decision | Reason |
|-------|----------|--------|
| Missing dest balances | Fill with 0 | Domain: new accounts have 0 balance |
| Duplicates | Remove | System retry artifacts, not real transactions |
| Zero amounts | Remove | Invalid transactions, not real payments |
| Outliers (amount) | Cap at 99.9th pct | Large txns = fraud signal; can't remove |
| Skewed amounts | log1p transform for EDA | Better distribution visualization |

**Features engineered (10 new columns):**

| Feature | Formula | Fraud Insight |
|---------|---------|---------------|
| zero_balance_after | newbalanceOrig == 0 | Strongest single predictor |
| amount_to_balance | amount / (oldbalanceOrg + 1) | Fraudsters spend all available funds |
| balance_diff_orig | oldbalanceOrg - newbalanceOrig | Magnitude of sender balance drop |
| balance_diff_dest | newbalanceDest - oldbalanceDest | Receiver gain |
| balance_mismatch | abs(balance_diff - amount) > 1 | Transaction inconsistency flag |
| is_extreme_amount | amount > 200,000 | Regulatory threshold flag |
| risk_score | Composite 0–100 | Multi-signal weighted score |
| risk_tier | cut(risk_score, 4 bins) | Low/Medium/High/Critical |
| hour_of_day | step % 24 | Time-based pattern analysis |
| day_of_week | (step // 24) % 7 | Weekend vs weekday analysis |

---

## Phase 4: Exploratory Data Analysis

**EDA structure:**
1. **Univariate:** Distributions of amount, type, hour, risk_score
2. **Bivariate:** Each feature vs isFraud — correlation and fraud rate
3. **Multivariate:** Correlation heatmap, feature interactions
4. **Fraud Pattern:** Time heatmaps, amount decile analysis, risk tier validation

**Top EDA findings:**
- zero_balance_after: 18× higher fraud rate than average
- CASH_OUT fraud rate: ~1.4% | TRANSFER fraud rate: ~0.8%
- Fraud amounts: median ~$350K vs legitimate median ~$75K
- Correlation heatmap: zero_balance_after most correlated with isFraud (0.32)
- Risk tier validation: Critical tier shows 12× base fraud rate

---

## Phase 5: SQL Analytics

**MySQL database:** fraud_detection_db.transactions  
**19 queries organized in 3 categories:**

1. **Business Analysis (10 queries):** KPI dashboard, fraud by type, time patterns, detection effectiveness, top cases
2. **Advanced SQL (9 queries):** CTEs, window functions (RANK, LAG/LEAD, running totals, PERCENT_RANK), stored procedures, views

---

## Phase 6: Machine Learning

**Feature selection rationale:**
- 14 features selected from 25 columns
- Excluded: nameOrig, nameDest (IDs — no predictive value)
- Excluded: isFlaggedFraud (system artifact, not independent signal)
- Excluded: risk_tier (derived from risk_score — redundant)

**Train-test split:**
- 80% train / 20% test
- Stratified by isFraud to maintain class proportions
- Random seed = 42 for reproducibility

**Model selection rationale:**

| Model | Why Included |
|-------|-------------|
| Logistic Regression | Baseline, calibrated probabilities, fast inference |
| Decision Tree | Explainable rules, max_depth=6 for business interpretation |
| Random Forest | Best performance, feature importance, robust to imbalance |

**Imbalance handling:** class_weight='balanced' — adjusts loss function weights automatically. Equivalent to oversampling minority class by weighting each fraud sample ~166× more.

**Primary metric: Recall** — business cost of missing fraud > cost of false alarm.

---

## Phase 7: Dashboard Design

**Power BI design principles:**
- Executive page: KPI cards visible in 5 seconds without scrolling
- Analyst page: drill-down capability for investigation
- Color code: Red = fraud/high risk, Blue = legitimate/low risk, Orange = warning
- Drill-through: Right-click any account → full transaction history

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Severe class imbalance (0.6% fraud) | class_weight='balanced' + Recall as primary metric |
| Outliers in amount column | Winsorization (cap, don't remove) + is_extreme_amount flag |
| Missing destination balances | Domain-informed fill with 0 (new accounts) |
| Making ML explainable | Feature importance + LR coefficients + DT rule extraction |
| SQL performance on aggregations | Compound index on (type, is_fraud, amount) |

---

## Key Learnings

1. **Domain knowledge > algorithm sophistication.** Understanding that PaySim fraud only occurs in CASH_OUT/TRANSFER made feature engineering 10× more effective.

2. **The metric you optimize is a business decision.** Recall vs Precision tradeoff was resolved by asking: what costs more — a missed fraud or a false alarm?

3. **Rule-based and ML approaches complement each other.** Simple rules (zero_balance_after = 1) catch obvious fraud; ML catches the subtle multi-signal cases.

4. **SQL can validate ML.** Writing Precision/Recall calculations directly in SQL gave a non-ML-dependent way to measure detection effectiveness.

5. **EDA findings directly inform features.** The correlation heatmap showing zero_balance_after as the strongest predictor directly led to it being the top feature in the ML model.
