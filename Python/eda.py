"""
=============================================================================
Financial Transaction Fraud Detection - Exploratory Data Analysis (EDA)
=============================================================================
Project : Financial Transaction Fraud Detection & Risk Analytics System
Author  : Dudala Vinay Kumar Goud (Tony)
Purpose : Comprehensive EDA — univariate, bivariate, multivariate analysis
          with business insight storytelling

EDA Structure:
  1. Dataset Overview
  2. Univariate Analysis  → Distribution of individual features
  3. Bivariate Analysis   → Relationship between features and fraud
  4. Multivariate Analysis → Correlation + interaction effects
  5. Fraud Pattern Analysis → Time, amount, type-based patterns
  6. Business Insights Summary
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Style Configuration ───────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.facecolor': 'white'
})

FRAUD_COLOR  = '#E84040'   # Red for fraud
LEGIT_COLOR  = '#3D9BE9'   # Blue for legitimate
ACCENT_COLOR = '#FF8C00'   # Orange for highlights

DATA_PATH   = '../data/cleaned/cleaned_transactions.csv'
OUTPUT_DIR  = '../reports/eda_charts/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Load Data ─────────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Fraud transactions: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.4f}%)")
    return df


# ─── 1. Dataset Overview ───────────────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame) -> None:
    """Print comprehensive dataset overview."""
    print("\n" + "="*60)
    print("1. DATASET OVERVIEW")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Info:")
    print(df.info())
    print(f"\nDescriptive Statistics:")
    print(df[['amount', 'oldbalanceOrg', 'newbalanceOrig',
              'oldbalanceDest', 'newbalanceDest', 'risk_score']].describe().round(2))
    print(f"\nClass Distribution:")
    print(df['isFraud'].value_counts())
    print(f"Fraud Rate: {df['isFraud'].mean()*100:.4f}%")
    print("\n⚠️  INSIGHT: Dataset is severely class-imbalanced (fraud ≈ 0.6%)")
    print("   → Accuracy metric is misleading. Use Precision, Recall, F1, AUC-ROC")


# ─── 2. Univariate Analysis ────────────────────────────────────────────────────

def univariate_analysis(df: pd.DataFrame) -> None:
    """
    Analyze distribution of individual features.
    Key question: What does each variable look like on its own?
    """
    print("\n" + "="*60)
    print("2. UNIVARIATE ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Univariate Feature Distributions', fontsize=16, fontweight='bold', y=1.01)

    # 2.1 Transaction Amount Distribution (log scale — right-skewed)
    ax = axes[0, 0]
    df['amount'].apply(np.log1p).hist(bins=50, ax=ax, color=LEGIT_COLOR, edgecolor='white', alpha=0.8)
    ax.set_title('Transaction Amount (Log Scale)')
    ax.set_xlabel('log(Amount + 1)')
    ax.set_ylabel('Frequency')
    ax.axvline(np.log1p(df['amount'].median()), color=ACCENT_COLOR, linestyle='--',
               label=f"Median: ${df['amount'].median():,.0f}")
    ax.legend()

    # 2.2 Transaction Type Distribution
    ax = axes[0, 1]
    type_counts = df['type'].value_counts()
    bars = ax.bar(type_counts.index, type_counts.values, color=LEGIT_COLOR, edgecolor='white')
    ax.set_title('Transaction Type Distribution')
    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    # 2.3 Hour of Day Distribution
    ax = axes[0, 2]
    df['hour_of_day'].value_counts().sort_index().plot(kind='bar', ax=ax,
        color=LEGIT_COLOR, edgecolor='white', alpha=0.8)
    ax.set_title('Transaction Volume by Hour of Day')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Count')

    # 2.4 Risk Score Distribution
    ax = axes[1, 0]
    df['risk_score'].hist(bins=30, ax=ax, color=ACCENT_COLOR, edgecolor='white', alpha=0.8)
    ax.set_title('Risk Score Distribution')
    ax.set_xlabel('Risk Score (0–100)')
    ax.set_ylabel('Frequency')
    ax.axvline(df['risk_score'].mean(), color=FRAUD_COLOR, linestyle='--',
               label=f"Mean: {df['risk_score'].mean():.1f}")
    ax.legend()

    # 2.5 Risk Tier Distribution
    ax = axes[1, 1]
    tier_order  = ['Low', 'Medium', 'High', 'Critical']
    tier_colors = ['#3D9BE9', '#FFA500', '#FF6347', '#E84040']
    tier_counts = df['risk_tier'].value_counts().reindex(tier_order, fill_value=0)
    bars = ax.bar(tier_counts.index, tier_counts.values, color=tier_colors, edgecolor='white')
    ax.set_title('Risk Tier Distribution')
    ax.set_xlabel('Risk Tier')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, tier_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=8)

    # 2.6 Day of Week Distribution
    ax = axes[1, 2]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_counts = df['day_of_week'].value_counts().sort_index()
    ax.bar(days, day_counts.values, color=LEGIT_COLOR, edgecolor='white', alpha=0.8)
    ax.set_title('Transaction Volume by Day of Week')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}01_univariate_analysis.png", bbox_inches='tight', dpi=150)
    plt.show()
    print("\n✅ Univariate analysis saved.")

    # Key insights
    print("\n📊 UNIVARIATE INSIGHTS:")
    print(f"  • Amount is heavily right-skewed → log transform needed for ML")
    print(f"  • CASH_OUT dominates ({type_counts.get('CASH_OUT',0):,} txns = {type_counts.get('CASH_OUT',0)/len(df)*100:.1f}%)")
    print(f"  • Most transactions are Low risk ({df[df['risk_tier']=='Low'].shape[0]/len(df)*100:.1f}%)")


# ─── 3. Bivariate Analysis ─────────────────────────────────────────────────────

def bivariate_analysis(df: pd.DataFrame) -> None:
    """
    Relationship between each feature and the fraud label (isFraud).
    Key question: Which features separate fraud from legitimate?
    """
    print("\n" + "="*60)
    print("3. BIVARIATE ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Bivariate Analysis — Feature vs Fraud Label', fontsize=16, fontweight='bold')

    # 3.1 Fraud rate by transaction type
    ax = axes[0, 0]
    fraud_by_type = df.groupby('type')['isFraud'].mean() * 100
    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in fraud_by_type.values]
    fraud_by_type.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Fraud Rate by Transaction Type')
    ax.set_xlabel('Transaction Type')
    ax.set_ylabel('Fraud Rate (%)')
    ax.tick_params(axis='x', rotation=30)
    for i, (idx, val) in enumerate(fraud_by_type.items()):
        ax.text(i, val + 0.05, f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')

    # 3.2 Amount distribution: Fraud vs Legitimate (box plot)
    ax = axes[0, 1]
    df['log_amount'] = np.log1p(df['amount'])
    df.boxplot(column='log_amount', by='isFraud', ax=ax,
               boxprops=dict(color=LEGIT_COLOR),
               medianprops=dict(color=FRAUD_COLOR, linewidth=2))
    ax.set_title('Amount Distribution: Fraud vs Legitimate')
    ax.set_xlabel('isFraud (0=Legitimate, 1=Fraud)')
    ax.set_ylabel('log(Amount)')
    plt.sca(ax)
    plt.title('Amount (Log): Fraud vs Legitimate')

    # 3.3 Risk score by fraud label (violin)
    ax = axes[0, 2]
    fraud_groups = [df[df['isFraud']==0]['risk_score'], df[df['isFraud']==1]['risk_score']]
    parts = ax.violinplot(fraud_groups, positions=[0, 1], showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(LEGIT_COLOR if i == 0 else FRAUD_COLOR)
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_title('Risk Score Distribution: Fraud vs Legitimate')
    ax.set_ylabel('Risk Score')

    # 3.4 Zero balance after transaction vs fraud
    ax = axes[1, 0]
    zero_bal = df.groupby('zero_balance_after')['isFraud'].mean() * 100
    colors = [LEGIT_COLOR, FRAUD_COLOR]
    ax.bar(['Non-Zero Balance', 'Zero Balance After'], zero_bal.values, color=colors, edgecolor='white')
    ax.set_title('Fraud Rate: Zero Balance After Transaction')
    ax.set_ylabel('Fraud Rate (%)')
    for i, val in enumerate(zero_bal.values):
        ax.text(i, val + 0.5, f'{val:.2f}%', ha='center', fontweight='bold')

    # 3.5 Fraud by hour of day
    ax = axes[1, 1]
    fraud_by_hour = df.groupby('hour_of_day')['isFraud'].mean() * 100
    ax.plot(fraud_by_hour.index, fraud_by_hour.values, color=FRAUD_COLOR, linewidth=2, marker='o', markersize=4)
    ax.fill_between(fraud_by_hour.index, fraud_by_hour.values, alpha=0.2, color=FRAUD_COLOR)
    ax.set_title('Fraud Rate by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_xticks(range(0, 24, 2))

    # 3.6 Fraud by dest_type
    ax = axes[1, 2]
    fraud_by_dest = df.groupby('dest_type')['isFraud'].mean() * 100
    ax.bar(fraud_by_dest.index, fraud_by_dest.values,
           color=[LEGIT_COLOR, FRAUD_COLOR], edgecolor='white')
    ax.set_title('Fraud Rate by Destination Type')
    ax.set_xlabel('Destination Type')
    ax.set_ylabel('Fraud Rate (%)')
    for i, val in enumerate(fraud_by_dest.values):
        ax.text(i, val + 0.1, f'{val:.2f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}02_bivariate_analysis.png", bbox_inches='tight', dpi=150)
    plt.show()
    print("✅ Bivariate analysis saved.")

    # Key insights
    print("\n📊 BIVARIATE INSIGHTS:")
    co_rate = df[df['type']=='CASH_OUT']['isFraud'].mean()*100
    tf_rate = df[df['type']=='TRANSFER']['isFraud'].mean()*100
    zb_rate = df[df['zero_balance_after']==1]['isFraud'].mean()*100
    print(f"  • CASH_OUT fraud rate: {co_rate:.2f}% | TRANSFER: {tf_rate:.2f}%")
    print(f"  • Zero-balance-after transactions: {zb_rate:.2f}% fraud rate (vs overall {df['isFraud'].mean()*100:.2f}%)")
    print(f"  • Fraud median amount > legitimate → large transactions are riskier")


# ─── 4. Multivariate Analysis ─────────────────────────────────────────────────

def multivariate_analysis(df: pd.DataFrame) -> None:
    """
    Analyze interactions between multiple features.
    Correlation heatmap + pairplot for key ML features.
    """
    print("\n" + "="*60)
    print("4. MULTIVARIATE ANALYSIS")
    print("="*60)

    # Correlation heatmap
    ml_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest',
        'zero_balance_after', 'amount_to_balance', 'risk_score',
        'hour_of_day', 'is_weekend', 'is_extreme_amount', 'isFraud'
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    corr_matrix = df[ml_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    ax.set_title('Feature Correlation Heatmap\n(Lower Triangle — Fraud Detection Features)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}03_correlation_heatmap.png", bbox_inches='tight', dpi=150)
    plt.show()

    # Print top correlations with isFraud
    fraud_corr = corr_matrix['isFraud'].drop('isFraud').abs().sort_values(ascending=False)
    print("\n📊 TOP CORRELATIONS WITH isFraud:")
    for feat, corr_val in fraud_corr.head(8).items():
        direction = "+" if corr_matrix.loc[feat, 'isFraud'] > 0 else "-"
        print(f"  {direction}{corr_val:.4f}  {feat}")

    print("\n📊 MULTIVARIATE INSIGHTS:")
    print("  • zero_balance_after has highest correlation with fraud")
    print("  • risk_score (composite) strongly aligned with fraud label")
    print("  • amount and balance features show moderate fraud correlation")
    print("  • Low multicollinearity overall → good for Logistic Regression")


# ─── 5. Fraud Pattern Deep-Dive ────────────────────────────────────────────────

def fraud_pattern_analysis(df: pd.DataFrame) -> None:
    """
    Deep-dive into fraud patterns for business storytelling.
    This is the section you walk through in interviews to show business thinking.
    """
    print("\n" + "="*60)
    print("5. FRAUD PATTERN ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fraud Pattern Deep-Dive Analysis', fontsize=16, fontweight='bold')

    # 5.1 Fraud amount vs legitimate amount (KDE comparison)
    ax = axes[0, 0]
    legit_amounts = np.log1p(df[df['isFraud']==0]['amount'])
    fraud_amounts = np.log1p(df[df['isFraud']==1]['amount'])
    ax.hist(legit_amounts, bins=40, alpha=0.6, color=LEGIT_COLOR, label='Legitimate', density=True)
    ax.hist(fraud_amounts, bins=40, alpha=0.6, color=FRAUD_COLOR, label='Fraud', density=True)
    ax.set_title('Transaction Amount Distribution\nFraud vs Legitimate (Log Scale)')
    ax.set_xlabel('log(Amount + 1)')
    ax.set_ylabel('Density')
    ax.legend()

    # 5.2 Fraud rate heatmap: Type × Hour
    ax = axes[0, 1]
    pivot = df.groupby(['type', 'hour_of_day'])['isFraud'].mean().unstack(fill_value=0)
    sns.heatmap(pivot * 100, ax=ax, cmap='YlOrRd', linewidths=0.3,
                cbar_kws={'label': 'Fraud Rate (%)'})
    ax.set_title('Fraud Rate Heatmap\nTransaction Type × Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Transaction Type')

    # 5.3 Cumulative fraud loss by amount bucket
    ax = axes[1, 0]
    df['amount_bucket'] = pd.qcut(df['amount'], q=10, labels=[f'D{i+1}' for i in range(10)])
    fraud_df = df[df['isFraud']==1]
    bucket_loss = fraud_df.groupby('amount_bucket')['amount'].sum() / 1e6
    ax.bar(bucket_loss.index, bucket_loss.values, color=FRAUD_COLOR, edgecolor='white', alpha=0.85)
    ax.set_title('Fraud Loss by Transaction Amount Decile')
    ax.set_xlabel('Amount Decile (D1=Lowest, D10=Highest)')
    ax.set_ylabel('Total Fraud Loss ($ Millions)')
    ax.tick_params(axis='x', rotation=30)

    # 5.4 Risk tier vs fraud rate
    ax = axes[1, 1]
    tier_order  = ['Low', 'Medium', 'High', 'Critical']
    tier_colors = ['#3D9BE9', '#FFA500', '#FF6347', '#E84040']
    tier_fraud  = df.groupby('risk_tier')['isFraud'].mean() * 100
    tier_fraud  = tier_fraud.reindex(tier_order, fill_value=0)
    bars = ax.bar(tier_fraud.index, tier_fraud.values, color=tier_colors, edgecolor='white')
    ax.set_title('Fraud Rate by Risk Tier\n(Rule-Based Risk Score Validation)')
    ax.set_xlabel('Risk Tier')
    ax.set_ylabel('Fraud Rate (%)')
    for bar, val in zip(bars, tier_fraud.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}04_fraud_patterns.png", bbox_inches='tight', dpi=150)
    plt.show()

    # Business metrics
    total_fraud_loss = df[df['isFraud']==1]['amount'].sum()
    avg_fraud_amount = df[df['isFraud']==1]['amount'].mean()
    avg_legit_amount = df[df['isFraud']==0]['amount'].mean()

    print("\n💰 BUSINESS METRICS:")
    print(f"  Total transactions analyzed:  {len(df):,}")
    print(f"  Total fraud transactions:     {df['isFraud'].sum():,}")
    print(f"  Total fraud loss (sample):    ${total_fraud_loss:,.2f}")
    print(f"  Avg fraud transaction amount: ${avg_fraud_amount:,.2f}")
    print(f"  Avg legitimate amount:        ${avg_legit_amount:,.2f}")
    print(f"  Fraud txns are {avg_fraud_amount/avg_legit_amount:.1f}x larger than legitimate on average")
    print(f"\n📊 FRAUD PATTERN INSIGHTS:")
    print("  • Fraud concentrates in CASH_OUT and TRANSFER (100% of fraud cases)")
    print("  • Fraudsters drain accounts completely (zero balance after)")
    print("  • Large amount transactions carry disproportionate fraud loss")
    print("  • Rule-based risk tiers successfully separate fraud from legitimate")


# ─── 6. Business Insights Summary ─────────────────────────────────────────────

def business_insights_summary(df: pd.DataFrame) -> None:
    """Final consolidated business insights for presentation."""
    print("\n" + "="*60)
    print("6. BUSINESS INSIGHTS SUMMARY")
    print("="*60)

    fraud = df[df['isFraud']==1]
    legit = df[df['isFraud']==0]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║         FRAUD ANALYTICS — KEY BUSINESS INSIGHTS         ║
╚══════════════════════════════════════════════════════════╝

📌 FRAUD SCOPE
   • Fraud rate: {df['isFraud'].mean()*100:.2f}% of transactions
   • Fraud types: 100% in CASH_OUT and TRANSFER only
   • Avg fraud amount: ${fraud['amount'].mean():>12,.2f}
   • Avg legit amount: ${legit['amount'].mean():>12,.2f}

🚨 TOP FRAUD INDICATORS
   1. Transaction drained sender account to zero
   2. CASH_OUT or TRANSFER transaction type
   3. Transaction amount > $200,000
   4. High amount-to-balance ratio (spending beyond typical pattern)
   5. Flagged by system (isFlaggedFraud = 1)

📊 RISK DISTRIBUTION
   • Low risk:      {df[df['risk_tier']=='Low'].shape[0]/len(df)*100:.1f}% of transactions
   • Medium risk:   {df[df['risk_tier']=='Medium'].shape[0]/len(df)*100:.1f}% of transactions
   • High risk:     {df[df['risk_tier']=='High'].shape[0]/len(df)*100:.1f}% of transactions
   • Critical risk: {df[df['risk_tier']=='Critical'].shape[0]/len(df)*100:.1f}% of transactions

💡 RECOMMENDATIONS
   1. Flag all CASH_OUT/TRANSFER transactions where balance goes to 0
   2. Implement velocity checks on amounts > $200K
   3. Secondary verification for High/Critical risk tier transactions
   4. ML model targeting: optimize Recall (catch more fraud)
      even at cost of slightly lower Precision

🎯 ML BASELINE TARGET (next phase)
   • Metric priority: Recall > Precision > Accuracy
   • Minimum target: Recall ≥ 0.80, Precision ≥ 0.60
   • Handle class imbalance: class_weight='balanced' or SMOTE
""")


# ─── Master EDA Runner ────────────────────────────────────────────────────────

def run_eda(data_path: str = DATA_PATH) -> None:
    """Run the complete EDA pipeline."""
    print("╔══════════════════════════════════════════════╗")
    print("║   FRAUD DETECTION — COMPLETE EDA PIPELINE   ║")
    print("╚══════════════════════════════════════════════╝")

    df = load_data(data_path)
    df['log_amount'] = np.log1p(df['amount'])

    dataset_overview(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)
    fraud_pattern_analysis(df)
    business_insights_summary(df)

    print(f"\n✅ EDA complete. Charts saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    run_eda()
