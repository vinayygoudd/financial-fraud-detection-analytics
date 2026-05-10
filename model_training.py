"""
=============================================================================
Financial Transaction Fraud Detection - ML Model Training & Evaluation
=============================================================================
Project : Financial Transaction Fraud Detection & Risk Analytics System
Author  : Dudala Vinay Kumar Goud (Tony)
Purpose : Train and evaluate interpretable ML models for fraud detection

Models:
  1. Logistic Regression  → Baseline, highly interpretable, fast
  2. Decision Tree        → Explainable rules, visualizable
  3. Random Forest        → Best accuracy, feature importance

Key considerations:
  - Severely imbalanced dataset (fraud ≈ 0.6%)
  - Primary metric: Recall (catch fraud), secondary: Precision
  - Interpretability > complexity (Junior DS positioning)
  - SMOTE for oversampling + class_weight='balanced'
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_PATH   = '../data/cleaned/cleaned_transactions.csv'
MODEL_DIR   = '../models/'
REPORTS_DIR = '../reports/ml_charts/'
RANDOM_SEED = 42

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 120, 'axes.titlesize': 13, 'axes.titleweight': 'bold'})

# ─── ML Feature Set ───────────────────────────────────────────────────────────
# These are the features we feed to all models
ML_FEATURES = [
    'amount',               # Raw transaction amount
    'oldbalanceOrg',        # Sender balance before
    'newbalanceOrig',       # Sender balance after
    'oldbalanceDest',       # Receiver balance before
    'newbalanceDest',       # Receiver balance after
    'balance_diff_orig',    # How much sender lost
    'balance_diff_dest',    # How much receiver gained
    'zero_balance_after',   # Strongest fraud signal
    'amount_to_balance',    # Ratio: amount / sender balance
    'balance_mismatch',     # Inconsistency flag
    'is_extreme_amount',    # Amount > $200K flag
    'hour_of_day',          # Time of transaction
    'is_weekend',           # Weekend flag
    'type_encoded',         # Transaction type (numeric)
]

TARGET = 'isFraud'


# ─── 1. Load & Prepare Data ───────────────────────────────────────────────────

def load_and_prepare(path: str = DATA_PATH):
    """
    Load cleaned data and prepare train/test split.

    CRITICAL DECISION: Stratified split
    → Ensures both train and test sets have proportional fraud samples.
      Without stratification, test set might have 0 fraud cases!
    """
    print("Loading cleaned dataset...")
    df = pd.read_csv(path)

    X = df[ML_FEATURES].copy()
    y = df[TARGET].copy()

    print(f"Features: {X.shape[1]} | Samples: {len(X):,}")
    print(f"Class distribution — 0: {(y==0).sum():,} | 1 (Fraud): {(y==1).sum():,}")
    print(f"Fraud rate: {y.mean()*100:.4f}%")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\nTrain: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

    # Feature scaling (required for Logistic Regression, optional for trees)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ─── 2. Model Evaluation Utility ─────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test, X_test_scaled=None) -> dict:
    """
    Comprehensive model evaluation — beyond just accuracy.

    For fraud detection, the metric priority is:
    Recall > F1 > Precision > Accuracy

    WHY? Missing a fraud (False Negative) costs the company money.
    A false alarm (False Positive) is just an annoying verification step.
    """
    use_scaled = X_test_scaled is not None
    X_eval = X_test_scaled if use_scaled else X_test

    y_pred  = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # Core metrics
    precision   = precision_score(y_test, y_pred, zero_division=0)
    recall      = recall_score(y_test, y_pred, zero_division=0)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_test, y_proba)
    avg_prec    = average_precision_score(y_test, y_proba)

    results = {
        'Model'         : name,
        'Precision'     : round(precision, 4),
        'Recall'        : round(recall, 4),
        'F1-Score'      : round(f1, 4),
        'ROC-AUC'       : round(roc_auc, 4),
        'Avg-Precision' : round(avg_prec, 4),
        'y_proba'       : y_proba,
        'y_pred'        : y_pred
    }

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}   ← Most important for fraud")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  ROC-AUC:         {roc_auc:.4f}")
    print(f"  Avg Precision:   {avg_prec:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    return results


# ─── 3. Logistic Regression ───────────────────────────────────────────────────

def train_logistic_regression(X_train_scaled, y_train) -> LogisticRegression:
    """
    Logistic Regression: Best for interpretability and probability calibration.

    class_weight='balanced': automatically adjusts weights inversely
    proportional to class frequencies. No need for SMOTE here.

    C=0.1: L2 regularization to prevent overfitting on sparse fraud class.

    Interview point: LR gives calibrated probabilities → great for fraud scoring.
    """
    print("\n▶ Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight='balanced',
        C=0.1,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    lr.fit(X_train_scaled, y_train)
    print("  ✅ Logistic Regression trained")
    return lr


# ─── 4. Decision Tree ─────────────────────────────────────────────────────────

def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Decision Tree: Fully explainable — you can show exact decision rules.
    max_depth=6: prevents overfitting while keeping interpretable structure.

    Interview point: DT can be printed as business rules for fraud investigators.
    E.g., "IF type=CASH_OUT AND amount>50000 AND balance_after=0 → FLAG"
    """
    print("\n▶ Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    dt.fit(X_train, y_train)
    print(f"  ✅ Decision Tree trained | Depth: {dt.get_depth()} | Leaves: {dt.get_n_leaves()}")
    return dt


# ─── 5. Random Forest ─────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Random Forest: Ensemble of decision trees → best accuracy in this stack.
    n_estimators=100: 100 trees balance performance and training time.
    class_weight='balanced_subsample': balances each bootstrap sample.

    Interview point: RF provides feature importance → tells business WHICH
    features drive fraud most, enabling better rule creation.
    """
    print("\n▶ Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced_subsample',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"  ✅ Random Forest trained | {rf.n_estimators} trees | OOB available")
    return rf


# ─── 6. ROC & PR Curve Comparison ─────────────────────────────────────────────

def plot_roc_pr_curves(models_results: list, y_test: pd.Series) -> None:
    """Plot ROC and Precision-Recall curves for all models side-by-side."""
    colors = ['#3D9BE9', '#FFA500', '#E84040']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison — ROC & Precision-Recall Curves', fontsize=14, fontweight='bold')

    # ROC Curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC=0.50)')
    for result, color in zip(models_results, colors):
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{result['Model']} (AUC={result['ROC-AUC']:.3f})")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve — Higher AUC = Better Model')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curve (better for imbalanced data)
    ax = axes[1]
    baseline_pr = y_test.mean()
    ax.axhline(baseline_pr, color='k', linestyle='--', lw=1,
               label=f'Random Baseline (P={baseline_pr:.3f})')
    for result, color in zip(models_results, colors):
        prec, rec, _ = precision_recall_curve(y_test, result['y_proba'])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{result['Model']} (AP={result['Avg-Precision']:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve\n(More informative for imbalanced data)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}05_roc_pr_curves.png", bbox_inches='tight', dpi=150)
    plt.show()
    print("✅ ROC/PR curves saved.")


# ─── 7. Confusion Matrix ──────────────────────────────────────────────────────

def plot_confusion_matrices(models_results: list, y_test: pd.Series) -> None:
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices — Fraud Detection Models', fontsize=14, fontweight='bold')

    for ax, result in zip(axes, models_results):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud']
        )
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{result['Model']}\n"
            f"Recall: {result['Recall']:.3f} | Precision: {result['Precision']:.3f}\n"
            f"FN (Missed Fraud): {fn} | FP (False Alarms): {fp}"
        )

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}06_confusion_matrices.png", bbox_inches='tight', dpi=150)
    plt.show()
    print("✅ Confusion matrices saved.")


# ─── 8. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(rf_model: RandomForestClassifier,
                            lr_model: LogisticRegression,
                            feature_names: list) -> None:
    """
    Feature importance from Random Forest + Logistic Regression coefficients.
    This is the BUSINESS INSIGHT slide — tells what drives fraud.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Feature Importance Analysis\n(What Drives Fraud?)', fontsize=14, fontweight='bold')

    # RF feature importance
    ax = axes[0]
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    colors = ['#E84040' if v > importances.mean() else '#3D9BE9' for v in importances.values]
    importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Random Forest — Feature Importance\n(Red = Above Average Importance)')
    ax.set_xlabel('Importance Score')
    ax.axvline(importances.mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
    ax.legend()

    # LR coefficients (absolute values)
    ax = axes[1]
    coef = pd.Series(np.abs(lr_model.coef_[0]), index=feature_names)
    coef = coef.sort_values(ascending=True)
    colors = ['#E84040' if v > coef.mean() else '#3D9BE9' for v in coef.values]
    coef.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Logistic Regression — Feature Coefficients\n(Absolute Value — Red = High Impact)')
    ax.set_xlabel('|Coefficient|')

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}07_feature_importance.png", bbox_inches='tight', dpi=150)
    plt.show()

    print("\n📊 TOP FRAUD DRIVERS (Random Forest):")
    top_features = importances.sort_values(ascending=False).head(5)
    for feat, imp in top_features.items():
        print(f"  {feat:30s}: {imp:.4f}")


# ─── 9. Model Comparison Summary ─────────────────────────────────────────────

def model_comparison_table(results: list) -> pd.DataFrame:
    """Create a clean comparison table of all models."""
    comparison = pd.DataFrame([{
        'Model'        : r['Model'],
        'Precision'    : r['Precision'],
        'Recall'       : r['Recall'],
        'F1-Score'     : r['F1-Score'],
        'ROC-AUC'      : r['ROC-AUC'],
        'Avg Precision': r['Avg-Precision'],
    } for r in results])

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(comparison.to_string(index=False))
    print("\n🏆 Best Recall (most important for fraud):",
          comparison.loc[comparison['Recall'].idxmax(), 'Model'])
    print("🏆 Best ROC-AUC:",
          comparison.loc[comparison['ROC-AUC'].idxmax(), 'Model'])
    return comparison


# ─── 10. Fraud Probability Scoring ───────────────────────────────────────────

def fraud_probability_scoring(rf_model: RandomForestClassifier,
                               X_test: pd.DataFrame,
                               y_test: pd.Series) -> pd.DataFrame:
    """
    Apply fraud probability score to each transaction.
    This is the DELIVERABLE for the fraud investigation team.

    A score of 0.85 means: 85% probability this transaction is fraud.
    Business teams use this to prioritize which transactions to review.
    """
    proba = rf_model.predict_proba(X_test)[:, 1]

    score_df = X_test.copy()
    score_df['fraud_probability'] = proba.round(4)
    score_df['fraud_label']       = (proba > 0.5).astype(int)
    score_df['actual_fraud']      = y_test.values
    score_df['alert_priority']    = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 0.85, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    print("\n📊 FRAUD PROBABILITY SCORING:")
    print(score_df['alert_priority'].value_counts())
    print(f"\nHigh/Critical alert accuracy:")
    high_crit = score_df[score_df['alert_priority'].isin(['High', 'Critical'])]
    if len(high_crit) > 0:
        print(f"  True fraud in High/Critical: {high_crit['actual_fraud'].sum()}/{len(high_crit)}"
              f" ({high_crit['actual_fraud'].mean()*100:.1f}%)")

    return score_df


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def run_ml_pipeline():
    """Run complete ML pipeline: Load → Train → Evaluate → Compare → Score."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FRAUD DETECTION — ML TRAINING PIPELINE            ║")
    print(f"║   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 1. Load data
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = load_and_prepare()

    # 2. Train models
    lr_model = train_logistic_regression(X_train_s, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 3. Evaluate all models
    results = []
    results.append(evaluate_model('Logistic Regression', lr_model, X_test, y_test, X_test_s))
    results.append(evaluate_model('Decision Tree',       dt_model, X_test, y_test))
    results.append(evaluate_model('Random Forest',       rf_model, X_test, y_test))

    # 4. Visualizations
    plot_roc_pr_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(rf_model, lr_model, ML_FEATURES)

    # 5. Comparison table
    comparison_df = model_comparison_table(results)

    # 6. Fraud scoring (using best model: RF)
    score_df = fraud_probability_scoring(rf_model, pd.DataFrame(X_test, columns=ML_FEATURES), y_test)

    # 7. Save models
    joblib.dump(rf_model, f"{MODEL_DIR}random_forest_fraud.pkl")
    joblib.dump(lr_model, f"{MODEL_DIR}logistic_regression_fraud.pkl")
    joblib.dump(dt_model, f"{MODEL_DIR}decision_tree_fraud.pkl")
    joblib.dump(scaler,   f"{MODEL_DIR}feature_scaler.pkl")
    print(f"\n✅ Models saved to {MODEL_DIR}")
    print("\n✅ ML pipeline complete!")
    return rf_model, lr_model, dt_model, comparison_df, score_df


if __name__ == '__main__':
    rf, lr, dt, comparison, scores = run_ml_pipeline()
