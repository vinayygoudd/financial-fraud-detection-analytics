"""
=============================================================================
Financial Transaction Fraud Detection - Data Cleaning Pipeline
=============================================================================
Project : Financial Transaction Fraud Detection & Risk Analytics System
Dataset : PaySim Synthetic Transaction Dataset
Author  : Dudala Vinay Kumar Goud (Tony)
Purpose : End-to-end data cleaning pipeline producing analysis-ready dataset
GitHub  : github.com/vinayygoudd
=============================================================================

INDUSTRY WORKFLOW:
  Raw CSV → Audit → Clean → Feature Engineer → Encode → Cleaned CSV

WHY THIS MATTERS FOR INTERVIEWS:
  Interviewers check if you understand *why* each cleaning step is done,
  not just that you know the pandas syntax. Every step here has a business reason.
=============================================================================
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# ─── Logging Configuration ────────────────────────────────────────────────────
# Industry standard: always log your pipeline so you can audit what happened
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_cleaning.log')
    ]
)
logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = '../data/raw/raw_transactions.csv'
CLEAN_DATA_PATH = '../data/cleaned/cleaned_transactions.csv'
FRAUD_TYPES     = ['CASH_OUT', 'TRANSFER']   # Only these can be fraud per PaySim domain knowledge
MAX_AMOUNT_CAP  = 10_000_000                  # $10M cap for outlier handling


# ─── Step 1: Load ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset with basic diagnostics.

    Business reason: Always inspect shape, dtypes, and a sample before touching data.
    Common beginner mistake: jumping straight into cleaning without understanding the data.
    """
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Dtypes:\n{df.dtypes}")
    return df


# ─── Step 2: Data Quality Audit ───────────────────────────────────────────────

def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    Full data quality audit BEFORE any cleaning.
    Always audit first — don't blindly clean.

    Returns dict of issues found, for downstream reporting.

    Interview tip: Mention you always run a quality audit before cleaning.
    This shows systematic thinking, not just code execution.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: DATA QUALITY AUDIT")
    logger.info("=" * 60)

    audit = {}

    # Missing values — which columns, how many, what % 
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_report = pd.DataFrame({'count': missing, 'pct': missing_pct})
    missing_report = missing_report[missing_report['count'] > 0]
    audit['missing_values'] = missing_report.to_dict()
    logger.info(f"Missing values:\n{missing_report}")

    # Duplicate rows
    dupe_count = df.duplicated().sum()
    audit['duplicates'] = int(dupe_count)
    logger.info(f"Duplicate rows: {dupe_count:,}")

    # Invalid amounts (zero or negative — should never happen in financial data)
    invalid_amounts = (df['amount'] <= 0).sum()
    audit['invalid_amounts'] = int(invalid_amounts)
    logger.info(f"Zero/negative amounts: {invalid_amounts}")

    # Extreme outliers in amount
    q99 = df['amount'].quantile(0.999)
    outliers = (df['amount'] > q99).sum()
    audit['amount_outliers_q999'] = int(outliers)
    logger.info(f"Amount outliers (>99.9th pct, >{q99:,.0f}): {outliers}")

    # Class imbalance (critical for fraud detection)
    fraud_rate = df['isFraud'].mean() * 100
    audit['fraud_rate_pct'] = round(fraud_rate, 4)
    logger.info(f"Fraud rate: {fraud_rate:.4f}% → SEVERELY IMBALANCED — needs SMOTE/class_weight")

    # Transaction type distribution
    audit['type_distribution'] = df['type'].value_counts().to_dict()
    logger.info(f"Transaction types:\n{df['type'].value_counts()}")

    # Fraud confined to CASH_OUT / TRANSFER (domain knowledge validation)
    fraud_by_type = df[df['isFraud'] == 1]['type'].value_counts()
    audit['fraud_by_type'] = fraud_by_type.to_dict()
    logger.info(f"Fraud by transaction type:\n{fraud_by_type}")

    logger.info("=" * 60)
    return audit


# ─── Step 3: Handle Missing Values ────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy explanation (critical for interviews):
    - oldbalanceDest / newbalanceDest → fill with 0.0
      REASON: In PaySim, destination accounts with no history have 0 balance.
              Filling with median would introduce false information.
    - All other numerics → fill with median (robust to outliers vs mean)
    - Categoricals → fill with mode

    Common beginner mistake: filling everything with mean (skewed by outliers)
    or dropping rows with missing values (loses fraud samples we can't afford to lose)
    """
    logger.info("STEP 3: Handling missing values...")
    total_before = df.isnull().sum().sum()

    # Domain-specific: destination balances → 0 for new accounts
    for col in ['oldbalanceDest', 'newbalanceDest']:
        n = df[col].isnull().sum()
        if n > 0:
            df[col] = df[col].fillna(0.0)
            logger.info(f"  '{col}': {n} nulls → filled with 0.0 (new account logic)")

    # Remaining numeric → median
    for col in df.select_dtypes(include=[np.number]).columns:
        n = df[col].isnull().sum()
        if n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  '{col}': {n} nulls → filled with median ({median_val:.4f})")

    # Categoricals → mode
    for col in df.select_dtypes(include=['object']).columns:
        n = df[col].isnull().sum()
        if n > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  '{col}': {n} nulls → filled with mode ('{mode_val}')")

    total_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {total_before} → {total_after}")
    return df


# ─── Step 4: Remove Duplicates ────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows.

    Business reason: Duplicate transactions could be caused by system retries,
    data ingestion errors, or ETL bugs. They inflate transaction counts and
    skew fraud rate calculations.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    logger.info(f"STEP 4: Duplicates removed: {removed} ({before:,} → {after:,} rows)")
    return df.reset_index(drop=True)


# ─── Step 5: Remove Invalid Amounts ───────────────────────────────────────────

def remove_invalid_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove transactions with zero or negative amounts.

    Business reason: Financial transactions must have a positive value.
    Zero-amount entries are likely system errors or test records.
    Negative amounts are data entry errors.
    """
    before = len(df)
    df = df[df['amount'] > 0]
    after = len(df)
    logger.info(f"STEP 5: Invalid amounts removed: {before - after} rows")
    return df.reset_index(drop=True)


# ─── Step 6: Outlier Treatment ────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap 'amount' at 99.9th percentile — DON'T remove outliers in fraud data.

    WHY NOT REMOVE? Large transactions can BE the fraud. Removing them would
    destroy exactly the signal we want to detect.

    Strategy: Cap + Flag (Winsorization)
    - amount_capped: capped value for ML models
    - is_extreme_amount: flag for analysis and rules

    Interview gold: Always explain WHY you chose capping over removal for fraud data.
    """
    cap_val = df['amount'].quantile(0.999)
    df['amount_capped'] = df['amount'].clip(upper=cap_val)

    # Flag extreme transactions (>$200K — common regulatory reporting threshold)
    df['is_extreme_amount'] = (df['amount'] > 200_000).astype(int)
    logger.info(f"STEP 6: Amount capped at 99.9th pct (${cap_val:,.2f}). Extreme flagged: {df['is_extreme_amount'].sum()}")
    return df


# ─── Step 7: Feature Engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analytical and ML-ready features from raw columns.

    PaySim domain knowledge:
    - step = 1 hour of simulated time (max 744 = 31 days)
    - Fraud signature: sender balance goes to 0 after transaction

    Features created:
    1. hour_of_day         → time-based fraud pattern analysis
    2. day_of_week         → weekend vs weekday fraud rates
    3. balance_diff_orig   → how much sender's balance dropped
    4. balance_diff_dest   → how much receiver's balance grew
    5. zero_balance_after  → STRONGEST fraud signal in PaySim
    6. amount_to_balance   → ratio of transaction to available funds
    7. balance_mismatch    → inconsistency between deducted amount and balance change
    8. dest_type           → Merchant (M) or Customer (C)
    9. risk_score          → rule-based composite risk score (0-100)
    """
    logger.info("STEP 7: Engineering features...")

    # Time features (step = 1 hour)
    df['hour_of_day'] = df['step'] % 24
    df['day_of_week'] = (df['step'] // 24) % 7
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # Balance change features
    df['balance_diff_orig'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).round(2)
    df['balance_diff_dest'] = (df['newbalanceDest'] - df['oldbalanceDest']).round(2)

    # Fraud pattern features
    df['zero_balance_after']   = (df['newbalanceOrig'] == 0).astype(int)
    df['amount_to_balance']    = (df['amount'] / (df['oldbalanceOrg'] + 1)).round(6)
    df['balance_mismatch']     = (
        np.abs(df['balance_diff_orig'] - df['amount']) > 1
    ).astype(int)

    # Destination entity type
    df['dest_type'] = np.where(df['nameDest'].str.startswith('M'), 'Merchant', 'Customer')

    # Composite rule-based risk score (useful for explanation in interviews)
    df['risk_score'] = (
        (df['zero_balance_after']  * 35) +
        (df['amount_to_balance'].clip(0, 1) * 25) +
        (df['is_extreme_amount']   * 20) +
        (df['type'].isin(FRAUD_TYPES).astype(int) * 15) +
        (df['isFlaggedFraud']      * 5)
    ).clip(0, 100).round(2)

    # Risk tier (for dashboarding)
    df['risk_tier'] = pd.cut(
        df['risk_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True
    ).astype(str)

    logger.info(f"  Features added: hour_of_day, day_of_week, is_weekend, balance features, risk_score, risk_tier")
    logger.info(f"  Risk tier distribution:\n{df['risk_tier'].value_counts()}")
    return df


# ─── Step 8: Encode Categoricals ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode transaction 'type' for ML models.
    Keep original string column for business analysis.
    Add numeric encoding for modeling.

    Note: We use label encoding here, not one-hot, because tree-based
    models (Decision Tree, Random Forest) handle ordinal encoding well.
    For Logistic Regression, you'd one-hot encode instead.
    """
    type_map = {
        'CASH_OUT' : 0,
        'PAYMENT'  : 1,
        'CASH_IN'  : 2,
        'TRANSFER' : 3,
        'DEBIT'    : 4
    }
    df['type_encoded'] = df['type'].map(type_map).astype(int)
    logger.info("STEP 8: Categorical encoding complete → 'type_encoded' added")
    return df


# ─── Step 9: Final Validation ─────────────────────────────────────────────────

def final_validation(df: pd.DataFrame) -> None:
    """
    Post-cleaning validation checks.
    Always validate AFTER cleaning — never assume your pipeline worked correctly.
    """
    logger.info("=" * 60)
    logger.info("STEP 9: FINAL VALIDATION")
    logger.info("=" * 60)
    assert df.isnull().sum().sum() == 0,           "❌ Nulls still present!"
    assert df.duplicated().sum() == 0,             "❌ Duplicates still present!"
    assert (df['amount'] > 0).all(),               "❌ Invalid amounts still present!"
    assert 'risk_score' in df.columns,             "❌ Feature engineering failed!"
    assert df['type_encoded'].isnull().sum() == 0, "❌ Encoding failed!"

    logger.info(f"✅ Rows:          {len(df):,}")
    logger.info(f"✅ Columns:       {df.shape[1]}")
    logger.info(f"✅ Missing:       0")
    logger.info(f"✅ Duplicates:    0")
    logger.info(f"✅ Fraud rows:    {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.4f}%)")
    logger.info(f"✅ Amount range:  ${df['amount'].min():,.2f} – ${df['amount'].max():,.2f}")
    logger.info("=" * 60)


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def run_cleaning_pipeline(raw_path: str = RAW_DATA_PATH,
                          clean_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """
    Orchestrates all cleaning steps in order.
    This is the single function a pipeline scheduler (Airflow, cron) would call.
    """
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  FRAUD DETECTION — DATA CLEANING PIPELINE           ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data(raw_path)
    _  = audit_data_quality(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_invalid_amounts(df)
    df = cap_outliers(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    final_validation(df)

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    logger.info(f"Cleaned dataset saved → {clean_path}")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df


if __name__ == '__main__':
    df_clean = run_cleaning_pipeline()
    print(f"\n✅ Cleaning pipeline complete. Final shape: {df_clean.shape}")
    print("\nSample (first 3 rows):")
    print(df_clean[['type','amount','isFraud','risk_score','risk_tier']].head(3))
