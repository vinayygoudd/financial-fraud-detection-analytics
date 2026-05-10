-- =============================================================================
-- Financial Transaction Fraud Detection - SQL Analysis Files
-- =============================================================================
-- Project : Financial Transaction Fraud Detection & Risk Analytics System
-- Author  : Dudala Vinay Kumar Goud (Tony)
-- Database: MySQL 8.0+
-- Purpose : Complete SQL analytics from schema creation to advanced queries
-- =============================================================================
-- FILE STRUCTURE:
--   PART 1 → Schema & Table Creation
--   PART 2 → Business Analysis Queries (KPIs, summaries)
--   PART 3 → Advanced SQL (Window Functions, CTEs, Rankings)
-- =============================================================================


-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 1: SCHEMA & TABLE CREATION
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS fraud_detection_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE fraud_detection_db;

-- -----------------------------------------------------------------------------
-- Table: transactions
-- Source: PaySim synthetic dataset (cleaned)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    id                  INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Surrogate key',
    step                INT NOT NULL               COMMENT 'Hour of simulation (1–744 = 31 days)',
    type                VARCHAR(20) NOT NULL        COMMENT 'Transaction type: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT',
    amount              DECIMAL(18, 2) NOT NULL    COMMENT 'Transaction amount in USD',
    name_orig           VARCHAR(20) NOT NULL        COMMENT 'Sender account ID (C = customer)',
    old_balance_orig    DECIMAL(18, 2) NOT NULL    COMMENT 'Sender balance before transaction',
    new_balance_orig    DECIMAL(18, 2) NOT NULL    COMMENT 'Sender balance after transaction',
    name_dest           VARCHAR(20) NOT NULL        COMMENT 'Receiver account ID (M = merchant, C = customer)',
    old_balance_dest    DECIMAL(18, 2) NOT NULL    COMMENT 'Receiver balance before transaction',
    new_balance_dest    DECIMAL(18, 2) NOT NULL    COMMENT 'Receiver balance after transaction',
    is_fraud            TINYINT(1) NOT NULL DEFAULT 0 COMMENT '1 = Fraud, 0 = Legitimate',
    is_flagged_fraud    TINYINT(1) NOT NULL DEFAULT 0 COMMENT '1 = System flagged, 0 = Not flagged',
    -- Engineered features
    hour_of_day         TINYINT NOT NULL           COMMENT 'Hour extracted from step (0–23)',
    day_of_week         TINYINT NOT NULL           COMMENT 'Day extracted from step (0=Mon, 6=Sun)',
    is_weekend          TINYINT(1) NOT NULL        COMMENT '1 = Weekend, 0 = Weekday',
    balance_diff_orig   DECIMAL(18, 2) NOT NULL    COMMENT 'Sender balance drop (old - new)',
    zero_balance_after  TINYINT(1) NOT NULL        COMMENT '1 if sender account drained to 0',
    amount_to_balance   DECIMAL(18, 6) NOT NULL    COMMENT 'amount / (old_balance_orig + 1)',
    balance_mismatch    TINYINT(1) NOT NULL        COMMENT '1 if amount != balance_diff_orig',
    dest_type           VARCHAR(10) NOT NULL        COMMENT 'Merchant or Customer',
    is_extreme_amount   TINYINT(1) NOT NULL        COMMENT '1 if amount > $200,000',
    risk_score          DECIMAL(5, 2) NOT NULL     COMMENT 'Rule-based risk score 0–100',
    risk_tier           VARCHAR(10) NOT NULL        COMMENT 'Low | Medium | High | Critical',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Indexes for query performance
    INDEX idx_type          (type),
    INDEX idx_is_fraud      (is_fraud),
    INDEX idx_risk_tier     (risk_tier),
    INDEX idx_step          (step),
    INDEX idx_amount        (amount),
    INDEX idx_zero_balance  (zero_balance_after),
    INDEX idx_hour          (hour_of_day)
) ENGINE=InnoDB
  COMMENT='PaySim synthetic financial transaction data — cleaned and feature-engineered';

-- Verify table created
DESCRIBE transactions;


-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 2: BUSINESS ANALYSIS QUERIES
-- ═══════════════════════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────────────────────
-- Q1: Overall Fraud KPI Dashboard
-- PURPOSE: Single-query executive summary of all key fraud metrics
-- WHERE USED: Power BI KPI card, management report
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                        AS total_transactions,
    SUM(is_fraud)                                   AS total_fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)        AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                    AS total_fraud_loss,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2)
                                                    AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN is_fraud = 0 THEN amount END), 2)
                                                    AS avg_legit_amount,
    SUM(is_flagged_fraud)                           AS system_flagged_count,
    ROUND(SUM(is_flagged_fraud) / NULLIF(SUM(is_fraud), 0) * 100, 2)
                                                    AS flag_coverage_pct
FROM transactions;
-- INTERVIEW: "How would you build a fraud KPI dashboard?"
-- Answer: Start with this query → feed into Power BI KPI cards


-- ─────────────────────────────────────────────────────────────────────────────
-- Q2: Fraud Rate and Loss by Transaction Type
-- PURPOSE: Identify which transaction types carry the most fraud risk
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    type                                                    AS transaction_type,
    COUNT(*)                                                AS total_count,
    SUM(is_fraud)                                           AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)                AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                            AS fraud_loss,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2)   AS avg_fraud_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)
          / NULLIF(SUM(amount), 0) * 100, 4)                AS fraud_loss_pct_of_total
FROM transactions
GROUP BY type
ORDER BY fraud_loss DESC;
-- KEY INSIGHT: CASH_OUT and TRANSFER will have all fraud (domain knowledge)


-- ─────────────────────────────────────────────────────────────────────────────
-- Q3: Fraud by Hour of Day — Time Pattern Analysis
-- PURPOSE: Detect peak fraud hours for real-time alerting rules
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    hour_of_day,
    COUNT(*)                                            AS total_transactions,
    SUM(is_fraud)                                       AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)            AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                        AS fraud_loss
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC
LIMIT 10;
-- INTERVIEW: "What SQL would you write to find the riskiest hours?"


-- ─────────────────────────────────────────────────────────────────────────────
-- Q4: Risk Tier Analysis — Fraud Concentration
-- PURPOSE: Validate that rule-based risk tiers correctly identify fraud
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    risk_tier,
    COUNT(*)                                            AS total_count,
    SUM(is_fraud)                                       AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)            AS fraud_rate_pct,
    ROUND(AVG(risk_score), 2)                           AS avg_risk_score,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                        AS fraud_loss
FROM transactions
GROUP BY risk_tier
ORDER BY FIELD(risk_tier, 'Critical', 'High', 'Medium', 'Low');
-- EXPECTED: Critical tier should have highest fraud_rate_pct
-- This validates our rule-based risk scoring system


-- ─────────────────────────────────────────────────────────────────────────────
-- Q5: Zero Balance Drain Analysis
-- PURPOSE: Quantify the "account drain" fraud pattern
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    zero_balance_after,
    COUNT(*)                                            AS transaction_count,
    SUM(is_fraud)                                       AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)            AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                               AS avg_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                        AS fraud_loss
FROM transactions
WHERE type IN ('CASH_OUT', 'TRANSFER')  -- Only fraud-eligible types
GROUP BY zero_balance_after
ORDER BY zero_balance_after DESC;
-- KEY INSIGHT: zero_balance_after=1 should show dramatically higher fraud rate


-- ─────────────────────────────────────────────────────────────────────────────
-- Q6: Top 20 Highest-Value Fraud Transactions
-- PURPOSE: Fraud investigation team workqueue — highest priority cases
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    id,
    step,
    type,
    ROUND(amount, 2)                AS amount,
    name_orig                       AS sender_account,
    name_dest                       AS receiver_account,
    ROUND(old_balance_orig, 2)      AS sender_balance_before,
    ROUND(new_balance_orig, 2)      AS sender_balance_after,
    risk_score,
    risk_tier,
    hour_of_day,
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC
LIMIT 20;
-- This becomes the "Fraud Investigation Queue" in Power BI


-- ─────────────────────────────────────────────────────────────────────────────
-- Q7: Fraud Loss Trend Over Time (Step/Hour)
-- PURPOSE: Identify fraud surges for time-series dashboard
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    step                                                    AS simulation_hour,
    COUNT(*)                                                AS total_transactions,
    SUM(is_fraud)                                           AS fraud_count,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                            AS fraud_loss,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)                AS fraud_rate_pct
FROM transactions
GROUP BY step
ORDER BY step
LIMIT 50;
-- Use for time-series line chart in Power BI


-- ─────────────────────────────────────────────────────────────────────────────
-- Q8: Weekend vs Weekday Fraud Comparison
-- PURPOSE: Operational insight — do fraud patterns differ on weekends?
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END  AS day_category,
    COUNT(*)                                                     AS total_transactions,
    SUM(is_fraud)                                                AS fraud_count,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)                     AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2)        AS avg_fraud_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss
FROM transactions
GROUP BY is_weekend;


-- ─────────────────────────────────────────────────────────────────────────────
-- Q9: Detection Effectiveness — System Flag vs Actual Fraud
-- PURPOSE: Measure how well the existing flag system catches real fraud
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    is_fraud,
    is_flagged_fraud,
    COUNT(*)  AS count,
    CASE
        WHEN is_fraud = 0 AND is_flagged_fraud = 0 THEN 'True Negative (Correct)'
        WHEN is_fraud = 1 AND is_flagged_fraud = 1 THEN 'True Positive (Caught!)'
        WHEN is_fraud = 1 AND is_flagged_fraud = 0 THEN 'False Negative (MISSED!)'
        WHEN is_fraud = 0 AND is_flagged_fraud = 1 THEN 'False Positive (False Alarm)'
    END AS outcome
FROM transactions
GROUP BY is_fraud, is_flagged_fraud
ORDER BY is_fraud DESC, is_flagged_fraud DESC;
-- INTERVIEW: "How would you calculate precision/recall in SQL?"
-- Precision = TP / (TP + FP)
-- Recall    = TP / (TP + FN)


-- ─────────────────────────────────────────────────────────────────────────────
-- Q10: Precision and Recall in SQL
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    SUM(CASE WHEN is_fraud = 1 AND is_flagged_fraud = 1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN is_fraud = 0 AND is_flagged_fraud = 1 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN is_fraud = 1 AND is_flagged_fraud = 0 THEN 1 ELSE 0 END) AS false_negatives,
    SUM(CASE WHEN is_fraud = 0 AND is_flagged_fraud = 0 THEN 1 ELSE 0 END) AS true_negatives,
    ROUND(
        SUM(CASE WHEN is_fraud = 1 AND is_flagged_fraud = 1 THEN 1 ELSE 0 END) /
        NULLIF(SUM(is_flagged_fraud), 0) * 100, 2
    ) AS precision_pct,
    ROUND(
        SUM(CASE WHEN is_fraud = 1 AND is_flagged_fraud = 1 THEN 1 ELSE 0 END) /
        NULLIF(SUM(is_fraud), 0) * 100, 2
    ) AS recall_pct
FROM transactions;


-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 3: ADVANCED SQL — Window Functions, CTEs, Ranking
-- ═══════════════════════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────────────────────
-- A1: CTE — High-Risk Transaction Identification
-- PURPOSE: Find transactions that meet 2+ fraud criteria simultaneously
-- CONCEPT: CTE breaks complex logic into readable named steps
-- ─────────────────────────────────────────────────────────────────────────────
WITH fraud_signals AS (
    SELECT
        id,
        type,
        amount,
        name_orig,
        risk_score,
        risk_tier,
        is_fraud,
        -- Count how many fraud signals are triggered
        (
            CASE WHEN zero_balance_after = 1      THEN 1 ELSE 0 END +
            CASE WHEN is_extreme_amount = 1        THEN 1 ELSE 0 END +
            CASE WHEN balance_mismatch = 1         THEN 1 ELSE 0 END +
            CASE WHEN type IN ('CASH_OUT','TRANSFER') THEN 1 ELSE 0 END +
            CASE WHEN amount_to_balance > 0.9      THEN 1 ELSE 0 END
        ) AS signal_count
    FROM transactions
),
high_risk AS (
    SELECT *
    FROM fraud_signals
    WHERE signal_count >= 3   -- 3+ signals = very high suspicion
)
SELECT
    risk_tier,
    signal_count,
    COUNT(*)                                        AS transactions,
    SUM(is_fraud)                                   AS confirmed_fraud,
    ROUND(SUM(is_fraud) / COUNT(*) * 100, 2)        AS fraud_rate_pct,
    ROUND(AVG(risk_score), 2)                       AS avg_risk_score
FROM high_risk
GROUP BY risk_tier, signal_count
ORDER BY signal_count DESC, fraud_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- A2: Window Function — Running Total of Fraud Loss by Hour
-- PURPOSE: Show cumulative fraud exposure over simulation time
-- CONCEPT: SUM() OVER() with ROWS BETWEEN for running total
-- ─────────────────────────────────────────────────────────────────────────────
WITH hourly_fraud AS (
    SELECT
        hour_of_day,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS hourly_fraud_loss,
        COUNT(CASE WHEN is_fraud = 1 THEN 1 END)           AS hourly_fraud_count
    FROM transactions
    GROUP BY hour_of_day
)
SELECT
    hour_of_day,
    ROUND(hourly_fraud_loss, 2)                             AS fraud_loss,
    hourly_fraud_count,
    ROUND(
        SUM(hourly_fraud_loss) OVER (ORDER BY hour_of_day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2
    )                                                       AS cumulative_fraud_loss,
    ROUND(
        AVG(hourly_fraud_loss) OVER (ORDER BY hour_of_day
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2
    )                                                       AS rolling_3hr_avg_loss
FROM hourly_fraud
ORDER BY hour_of_day;
-- INTERVIEW: "What is a window function and when would you use it?"
-- Answer: A function that operates across a set of rows related to the current row
--         WITHOUT collapsing them into a single output row (unlike GROUP BY)


-- ─────────────────────────────────────────────────────────────────────────────
-- A3: Window Function — Ranking Accounts by Fraud Amount
-- PURPOSE: Identify which sender accounts are responsible for most fraud
-- CONCEPT: RANK() vs DENSE_RANK() vs ROW_NUMBER()
-- ─────────────────────────────────────────────────────────────────────────────
WITH sender_fraud AS (
    SELECT
        name_orig                                           AS sender_account,
        COUNT(*)                                            AS total_transactions,
        SUM(is_fraud)                                       AS fraud_transactions,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)
                                                            AS total_fraud_loss,
        ROUND(AVG(risk_score), 2)                           AS avg_risk_score
    FROM transactions
    WHERE is_fraud = 1
    GROUP BY name_orig
)
SELECT
    sender_account,
    total_transactions,
    fraud_transactions,
    total_fraud_loss,
    avg_risk_score,
    RANK()       OVER (ORDER BY total_fraud_loss DESC)  AS loss_rank,
    DENSE_RANK() OVER (ORDER BY total_fraud_loss DESC)  AS loss_dense_rank,
    NTILE(4)     OVER (ORDER BY total_fraud_loss DESC)  AS loss_quartile
FROM sender_fraud
ORDER BY total_fraud_loss DESC
LIMIT 20;
-- INTERVIEW: RANK() skips numbers after ties; DENSE_RANK() does not; ROW_NUMBER() is always unique


-- ─────────────────────────────────────────────────────────────────────────────
-- A4: Window Function — Percentile Ranking of Transaction Amounts
-- PURPOSE: Identify extreme transactions relative to all transactions
-- CONCEPT: PERCENT_RANK() and CUME_DIST() window functions
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    id,
    type,
    ROUND(amount, 2)                                    AS amount,
    is_fraud,
    risk_tier,
    ROUND(PERCENT_RANK() OVER (ORDER BY amount) * 100, 2) AS amount_percentile,
    ROUND(CUME_DIST()    OVER (ORDER BY amount) * 100, 2) AS cumulative_distribution
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────────────
-- A5: LAG/LEAD — Consecutive Fraud Pattern Detection
-- PURPOSE: Detect accounts with multiple consecutive fraud attempts
-- CONCEPT: LAG() to look at previous transaction from same account
-- ─────────────────────────────────────────────────────────────────────────────
WITH account_transactions AS (
    SELECT
        name_orig,
        step,
        type,
        amount,
        is_fraud,
        risk_score,
        -- Previous transaction details for the SAME account
        LAG(is_fraud) OVER (PARTITION BY name_orig ORDER BY step)  AS prev_is_fraud,
        LAG(amount)   OVER (PARTITION BY name_orig ORDER BY step)  AS prev_amount,
        LAG(step)     OVER (PARTITION BY name_orig ORDER BY step)  AS prev_step
    FROM transactions
),
repeat_fraud AS (
    SELECT
        name_orig,
        step,
        type,
        amount,
        is_fraud,
        prev_is_fraud,
        prev_amount,
        step - prev_step                                            AS hours_since_last_txn
    FROM account_transactions
    WHERE is_fraud = 1 AND prev_is_fraud = 1   -- Current AND previous both fraud
)
SELECT *
FROM repeat_fraud
ORDER BY hours_since_last_txn ASC
LIMIT 10;
-- INTERVIEW: LAG() accesses previous row; LEAD() accesses next row
-- Partition BY ensures we compare within the same account, not across all accounts


-- ─────────────────────────────────────────────────────────────────────────────
-- A6: Subquery + Aggregation — Above-Average Fraud Transactions
-- PURPOSE: Find transactions with amounts above the average fraud amount
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    t.id,
    t.type,
    ROUND(t.amount, 2)  AS amount,
    t.risk_tier,
    t.risk_score,
    ROUND(avg_fraud.avg_fraud_amount, 2)    AS avg_fraud_amount,
    ROUND(t.amount / avg_fraud.avg_fraud_amount, 2) AS times_above_avg
FROM transactions t
CROSS JOIN (
    SELECT AVG(amount) AS avg_fraud_amount
    FROM transactions
    WHERE is_fraud = 1
) avg_fraud
WHERE t.is_fraud = 1
  AND t.amount > avg_fraud.avg_fraud_amount
ORDER BY t.amount DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────────────
-- A7: Stored Procedure — Fraud Summary Report Generator
-- PURPOSE: Reusable report generation for any date range
-- CONCEPT: Stored procedures for parameterized business reports
-- ─────────────────────────────────────────────────────────────────────────────
DELIMITER //

CREATE PROCEDURE IF NOT EXISTS generate_fraud_report(
    IN p_risk_tier VARCHAR(10),      -- 'Low', 'Medium', 'High', 'Critical', or 'ALL'
    IN p_fraud_only TINYINT          -- 1 = fraud only, 0 = all transactions
)
BEGIN
    -- Fraud summary for a given risk tier and fraud filter
    SELECT
        risk_tier,
        type,
        COUNT(*)                                                    AS total_count,
        SUM(is_fraud)                                               AS fraud_count,
        ROUND(SUM(is_fraud) / COUNT(*) * 100, 4)                    AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss,
        ROUND(AVG(amount), 2)                                       AS avg_amount,
        ROUND(AVG(risk_score), 2)                                   AS avg_risk_score
    FROM transactions
    WHERE (p_risk_tier = 'ALL' OR risk_tier = p_risk_tier)
      AND (p_fraud_only = 0 OR is_fraud = p_fraud_only)
    GROUP BY risk_tier, type
    ORDER BY fraud_loss DESC;
END //

DELIMITER ;

-- Usage examples:
-- CALL generate_fraud_report('Critical', 0);  -- All Critical tier transactions
-- CALL generate_fraud_report('ALL', 1);        -- All fraud across all tiers
-- CALL generate_fraud_report('High', 1);       -- Fraud in High tier only


-- ─────────────────────────────────────────────────────────────────────────────
-- A8: View — Fraud Investigation Dashboard View
-- PURPOSE: Simplified view for fraud analyst team (hide complex joins)
-- CONCEPT: Views abstract query complexity for non-technical users
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW vw_fraud_investigation AS
SELECT
    id                                          AS case_id,
    step                                        AS simulation_hour,
    type                                        AS transaction_type,
    ROUND(amount, 2)                            AS transaction_amount,
    name_orig                                   AS sender_account,
    name_dest                                   AS receiver_account,
    ROUND(old_balance_orig, 2)                  AS sender_balance_before,
    ROUND(new_balance_orig, 2)                  AS sender_balance_after,
    dest_type                                   AS receiver_type,
    risk_score,
    risk_tier,
    hour_of_day,
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_category,
    CASE WHEN zero_balance_after = 1 THEN 'YES - Account Drained' ELSE 'No' END
                                                AS account_drained,
    CASE WHEN balance_mismatch = 1 THEN 'YES - Suspicious' ELSE 'Normal' END
                                                AS balance_consistency,
    is_fraud                                    AS confirmed_fraud,
    is_flagged_fraud                            AS system_flagged
FROM transactions
WHERE risk_tier IN ('High', 'Critical')   -- Analyst sees only high-priority cases
ORDER BY risk_score DESC;

-- Usage: SELECT * FROM vw_fraud_investigation LIMIT 100;


-- ─────────────────────────────────────────────────────────────────────────────
-- A9: Index Analysis — Query Optimization
-- PURPOSE: Show understanding of database performance optimization
-- ─────────────────────────────────────────────────────────────────────────────

-- Check existing indexes
SHOW INDEX FROM transactions;

-- Compound index for common fraud query pattern
CREATE INDEX IF NOT EXISTS idx_type_fraud_amount
ON transactions (type, is_fraud, amount);

-- EXPLAIN plan for fraud query (how MySQL executes it)
EXPLAIN SELECT type, COUNT(*), SUM(is_fraud)
FROM transactions
WHERE is_fraud = 1
GROUP BY type;
-- INTERVIEW: "How would you optimize a slow SQL query?"
-- Answer: EXPLAIN plan → check for full table scans → add indexes on WHERE/JOIN columns


-- ─────────────────────────────────────────────────────────────────────────────
-- A10: Final Business Summary Query
-- PURPOSE: One comprehensive query for the executive presentation
-- ─────────────────────────────────────────────────────────────────────────────
WITH kpis AS (
    SELECT
        COUNT(*)                                                AS total_txns,
        SUM(is_fraud)                                           AS fraud_txns,
        SUM(is_flagged_fraud)                                   AS flagged_txns,
        ROUND(SUM(is_fraud)/COUNT(*)*100, 4)                    AS fraud_rate,
        ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) AS total_fraud_loss,
        ROUND(AVG(CASE WHEN is_fraud=1 THEN amount END), 2)     AS avg_fraud_txn,
        MAX(CASE WHEN is_fraud=1 THEN amount ELSE 0 END)        AS max_fraud_txn,
        SUM(CASE WHEN risk_tier='Critical' THEN 1 ELSE 0 END)   AS critical_risk_count,
        ROUND(SUM(CASE WHEN risk_tier='Critical' AND is_fraud=1 THEN 1 ELSE 0 END)
              / NULLIF(SUM(CASE WHEN risk_tier='Critical' THEN 1 ELSE 0 END), 0) * 100, 2)
                                                                AS critical_fraud_rate
    FROM transactions
)
SELECT
    total_txns              AS `Total Transactions`,
    fraud_txns              AS `Fraud Transactions`,
    flagged_txns            AS `System Flagged`,
    CONCAT(fraud_rate, '%') AS `Fraud Rate`,
    CONCAT('$', FORMAT(total_fraud_loss, 2)) AS `Total Fraud Loss`,
    CONCAT('$', FORMAT(avg_fraud_txn, 2))    AS `Avg Fraud Amount`,
    CONCAT('$', FORMAT(max_fraud_txn, 2))    AS `Largest Fraud`,
    critical_risk_count     AS `Critical Risk Count`,
    CONCAT(critical_fraud_rate, '%') AS `Critical Tier Fraud Rate`
FROM kpis;

-- =============================================================================
-- END OF SQL ANALYSIS FILE
-- =============================================================================
