"""
STEP 1 OF 2 - THE DEVELOPER
===========================
In this step we play the MODEL DEVELOPER.

Goal: build a Probability-of-Default (PD) model. For every credit-card
customer it estimates the chance (0%-100%) that the customer will MISS next
month's payment (this is what "default" means in this dataset).

What the script does, in plain English:
  1. Load the data.
  2. Split it into two piles:
       - DEVELOPMENT sample (70%): the model is allowed to learn from this.
       - HOLD-OUT sample (30%): kept hidden; used only to test the model.
  3. Turn raw columns into useful model inputs ("features").
  4. Fit a logistic regression (a standard model for yes/no outcomes that
     outputs a probability).
  5. Save the model description and its predictions, so that a separate
     person (the validator, Step 2) can check the work.

Run it with:   python 01_develop_pd_model.py
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42  # fixed so that anyone re-running gets the same split and numbers
os.makedirs("outputs", exist_ok=True)

# Column groups in the raw data (each has 6 months of history: 1 = latest month)
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]        # repayment status per month
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]  # amount billed per month
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]  # amount paid per month


# ---------------------------------------------------------------------------
# Feature engineering: raw columns -> model inputs
# ---------------------------------------------------------------------------
def build_features(df):
    """Create the model inputs. Each line is one documented feature."""
    X = pd.DataFrame(index=df.index)

    # Repayment status of the most recent month.
    # -2 = no spending, -1 = paid in full, 0 = paid minimum,
    # 1..9 = number of months the payment is late.
    X["PAY_1"] = df["PAY_1"]

    # The worst repayment status seen in the last 6 months.
    X["MAX_DELINQ_6M"] = df[PAY_COLS].max(axis=1)

    # How many of the last 6 months the customer paid late.
    X["MONTHS_LATE_6M"] = (df[PAY_COLS] > 0).sum(axis=1)

    # Utilisation = average bill as a share of the credit limit.
    # Capped at 1.5 so a few extreme values cannot dominate.
    X["AVG_UTILISATION"] = (df[BILL_COLS].mean(axis=1) / df["LIMIT_BAL"]).clip(0, 1.5)

    # Payment ratio = how much of what was billed was actually paid (0 to 1).
    total_bill = df[BILL_COLS].clip(lower=0).sum(axis=1)
    total_paid = df[PAYAMT_COLS].sum(axis=1)
    ratio = total_paid / total_bill.replace(0, np.nan)   # avoid dividing by zero
    X["PAYMENT_RATIO"] = ratio.fillna(1.0).clip(0, 1)     # no bill -> nothing owed

    # Credit limit on a log scale (limits are very skewed).
    X["LOG_LIMIT"] = np.log(df["LIMIT_BAL"])

    X["AGE"] = df["AGE"]

    # Education as yes/no flags. "University" is the reference group.
    X["EDU_GRADUATE"] = (df["EDUCATION"] == 1).astype(int)
    X["EDU_HIGHSCHOOL"] = (df["EDUCATION"] == 3).astype(int)
    X["EDU_OTHER"] = df["EDUCATION"].isin([0, 4, 5, 6]).astype(int)

    # NOTE: SEX and MARRIAGE are deliberately NOT used. Lenders are not allowed
    # to price credit risk on these in many jurisdictions (fair lending).
    return X


# ---------------------------------------------------------------------------
# 1. Load data and split it
# ---------------------------------------------------------------------------
df = pd.read_csv("data/credit.csv")
y = df["default"]

dev_idx, hold_idx = train_test_split(
    df.index, test_size=0.30, stratify=y, random_state=SEED
)
sample_label = pd.Series("holdout", index=df.index)
sample_label.loc[dev_idx] = "development"
sample_label.rename("sample").to_csv("outputs/sample_split.csv", index_label="row")

print(f"Rows: {len(df):,} | development: {len(dev_idx):,} | hold-out: {len(hold_idx):,}")
print(f"Default rate  development: {y.loc[dev_idx].mean():.2%}  hold-out: {y.loc[hold_idx].mean():.2%}")

# ---------------------------------------------------------------------------
# 2. Features, scaled using the DEVELOPMENT sample only
# ---------------------------------------------------------------------------
X = build_features(df)

# Standardise: subtract the mean and divide by the standard deviation, so every
# input is on a comparable scale (coefficients then mean "per 1 std-dev change").
mean = X.loc[dev_idx].mean()
std = X.loc[dev_idx].std(ddof=0)
X_scaled = (X - mean) / std

# ---------------------------------------------------------------------------
# 3. Fit the logistic regression on the development sample
# ---------------------------------------------------------------------------
X_design = sm.add_constant(X_scaled, has_constant="add")  # adds the intercept
model = sm.Logit(y.loc[dev_idx], X_design.loc[dev_idx]).fit(disp=False)

print("\n" + str(model.summary()))
with open("outputs/developer_model_summary.txt", "w") as f:
    f.write(str(model.summary()))

# ---------------------------------------------------------------------------
# 4. Predict a PD for every customer and report the developer's own results
# ---------------------------------------------------------------------------
pd_hat = model.predict(X_design)

out = pd.DataFrame({"sample": sample_label, "default": y, "pd_model": pd_hat})
out.to_csv("outputs/developer_predictions.csv", index_label="row")

auc_dev = roc_auc_score(y.loc[dev_idx], pd_hat.loc[dev_idx])
auc_hold = roc_auc_score(y.loc[hold_idx], pd_hat.loc[hold_idx])
print(f"\nDeveloper's own claim -> AUC development: {auc_dev:.3f} | hold-out: {auc_hold:.3f}")

# ---------------------------------------------------------------------------
# 5. Save the model "documentation" the validator will work from
# ---------------------------------------------------------------------------
spec = {
    "model_type": "Logistic regression (statsmodels Logit) on standardised features",
    "target": "default (missed next month's payment)",
    "features": list(X.columns),
    "scaling_mean": mean.to_dict(),
    "scaling_std": std.to_dict(),
    "coefficients": model.params.to_dict(),
    "developer_auc_development": auc_dev,
    "developer_auc_holdout": auc_hold,
}
with open("outputs/developer_model_spec.json", "w") as f:
    json.dump(spec, f, indent=2)
print("Saved model spec and predictions to the outputs/ folder.")
