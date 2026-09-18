"""
STEP 2 OF 2 - THE VALIDATOR
===========================
In this step we play the INDEPENDENT VALIDATOR: the "examiner" who checks the
developer's model. The validator does NOT reuse the developer's code. It only
receives (a) the raw data, (b) the model documentation (developer_model_spec.json)
and (c) the developer's predictions, then does its own work.

The validator asks the questions a bank's model-risk team would ask:

  A. DATA          Is the data clean and sensible?
  B. REPLICATION   Can I rebuild the model from the documentation and get the
                   same answers?  (If not, the documentation or code is wrong.)
  C. LOGIC         Do the model's coefficients make business sense?
  D. RANKING       Does it put risky customers above safe ones?     (discrimination)
  E. ACCURACY      Are the predicted probabilities close to reality? (calibration)
  F. STABILITY     Does the data the model sees stay similar?        (PSI)
  G. BENCHMARK     Is it better than simple alternatives, and is a more
                   complex "challenger" model much better?
  H. SENSITIVITY   What happens to the predicted PDs if conditions worsen?

Every result is saved in outputs/validation/ and used to write validation_report.md.

Run it with:   python 02_validate_pd_model.py    (run Step 1 first)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42
OUT = "outputs/validation"
os.makedirs(OUT, exist_ok=True)
results = {}  # key numbers, saved at the end as key_results.json


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Load what the developer handed over
# ---------------------------------------------------------------------------
raw = pd.read_csv("data/credit.csv")
spec = json.load(open("outputs/developer_model_spec.json"))
dev_pred = pd.read_csv("outputs/developer_predictions.csv", index_col="row")
is_dev = (dev_pred["sample"] == "development").values
is_hold = ~is_dev
y = raw["default"].values


# ---------------------------------------------------------------------------
# The validator's OWN version of the feature-building code
# (written separately from the developer's, from the documentation only)
# ---------------------------------------------------------------------------
def rebuild_features(df):
    feats = pd.DataFrame(index=df.index)
    pay = df[[f"PAY_{m}" for m in range(1, 7)]]
    bill = df[[f"BILL_AMT{m}" for m in range(1, 7)]]
    paid = df[[f"PAY_AMT{m}" for m in range(1, 7)]]

    feats["PAY_1"] = df["PAY_1"]
    feats["MAX_DELINQ_6M"] = pay.max(axis=1)
    feats["MONTHS_LATE_6M"] = pay.gt(0).sum(axis=1)
    feats["AVG_UTILISATION"] = np.minimum(bill.mean(axis=1) / df["LIMIT_BAL"], 1.5)
    feats["AVG_UTILISATION"] = np.maximum(feats["AVG_UTILISATION"], 0)

    positive_bill = bill.clip(lower=0).sum(axis=1)
    ratio = np.where(positive_bill > 0, paid.sum(axis=1) / positive_bill.where(positive_bill > 0, 1), 1.0)
    feats["PAYMENT_RATIO"] = np.clip(ratio, 0, 1)

    feats["LOG_LIMIT"] = np.log(df["LIMIT_BAL"])
    feats["AGE"] = df["AGE"]
    feats["EDU_GRADUATE"] = (df["EDUCATION"] == 1).astype(int)
    feats["EDU_HIGHSCHOOL"] = (df["EDUCATION"] == 3).astype(int)
    feats["EDU_OTHER"] = df["EDUCATION"].isin([0, 4, 5, 6]).astype(int)
    return feats[spec["features"]]


def gini(auc):
    return 2 * auc - 1  # Gini is just AUC rescaled so that 0 = random, 1 = perfect


def ks_stat(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))  # biggest gap between "bad" and "good" customers


def metrics(y_true, score):
    auc = roc_auc_score(y_true, score)
    return {"AUC": auc, "Gini": gini(auc), "KS": ks_stat(y_true, score)}


# ===========================================================================
# A. DATA QUALITY
# ===========================================================================
heading("A. DATA QUALITY")
dq = [
    ("Rows", len(raw)),
    ("Missing values (all columns)", int(raw.isna().sum().sum())),
    ("Exact duplicate rows", int(raw.duplicated().sum())),
    ("EDUCATION codes outside documented 1-4 (0, 5, 6)", int((~raw["EDUCATION"].isin([1, 2, 3, 4])).sum())),
    ("MARRIAGE code 0 (undocumented)", int((raw["MARRIAGE"] == 0).sum())),
    ("PAY_1 values of -2 or 0 (not in original data dictionary)", int(raw["PAY_1"].isin([-2, 0]).sum())),
    ("Negative BILL_AMT values (credit balances/refunds)", int((raw[[f"BILL_AMT{m}" for m in range(1, 7)]] < 0).any(axis=1).sum())),
    ("Overall default rate", round(float(raw["default"].mean()), 4)),
]
dq_df = pd.DataFrame(dq, columns=["check", "result"])
print(dq_df.to_string(index=False))
dq_df.to_csv(f"{OUT}/A_data_quality.csv", index=False)

# ===========================================================================
# B. REPLICATION - rebuild the model from the documentation
# ===========================================================================
heading("B. REPLICATION")
X_val = rebuild_features(raw)
mean = pd.Series(spec["scaling_mean"])[spec["features"]]
std = pd.Series(spec["scaling_std"])[spec["features"]]

# (1) Do my re-built features, scaled with the documented mean/std, reproduce the developer's PDs
#     when I plug in the developer's documented coefficients?
coef = pd.Series(spec["coefficients"])
z = coef["const"] + ((X_val - mean) / std).dot(coef[spec["features"]])
pd_from_spec = 1 / (1 + np.exp(-z))
diff_spec = np.abs(pd_from_spec.values - dev_pred["pd_model"].values)

# (2) Independently RE-FIT the model with a different software library (scikit-learn)
#     on the development sample and compare.
X_scaled_val = (X_val - mean) / std
refit = LogisticRegression(C=1e8, max_iter=10000, tol=1e-10)  # C huge = no regularisation
refit.fit(X_scaled_val[is_dev], y[is_dev])
pd_refit = refit.predict_proba(X_scaled_val)[:, 1]
diff_refit = np.abs(pd_refit - dev_pred["pd_model"].values)

coef_refit = pd.Series(refit.coef_[0], index=spec["features"])
coef_dev = coef[spec["features"]]
replication = pd.DataFrame(
    {
        "check": [
            "Max |PD difference|: documented coefficients vs developer predictions",
            "Max |PD difference|: independent re-fit vs developer predictions",
            "Mean |PD difference|: independent re-fit vs developer predictions",
            "Max |coefficient difference|: re-fit vs developer",
        ],
        "result": [
            float(diff_spec.max()),
            float(diff_refit.max()),
            float(diff_refit.mean()),
            float(np.abs(coef_refit - coef_dev).max()),
        ],
    }
)
print(replication.to_string(index=False))
replication.to_csv(f"{OUT}/B_replication.csv", index=False)
results["replication_max_pd_diff_refit"] = float(diff_refit.max())
results["replication_max_pd_diff_spec"] = float(diff_spec.max())

# ===========================================================================
# C. CONCEPTUAL SOUNDNESS - do the signs and the inputs make sense?
# ===========================================================================
heading("C. CONCEPTUAL SOUNDNESS (coefficient signs and multicollinearity)")
# +1 means "a higher value should raise the risk of default", -1 the opposite, 0 = no strong prior.
expected_sign = {
    "PAY_1": 1, "MAX_DELINQ_6M": 1, "MONTHS_LATE_6M": 1, "AVG_UTILISATION": 1,
    "PAYMENT_RATIO": -1, "LOG_LIMIT": -1, "AGE": 0, "EDU_GRADUATE": 0,
    "EDU_HIGHSCHOOL": 0, "EDU_OTHER": 0,
}
rows = []
for f in spec["features"]:
    c = float(coef_dev[f])
    exp = expected_sign[f]
    if exp == 0:
        verdict = "no prior view"
    else:
        verdict = "OK" if np.sign(c) == exp else "SIGN CONFLICT"
    rows.append([f, round(c, 4), {1: "+", -1: "-", 0: "?"}[exp], verdict])
signs = pd.DataFrame(rows, columns=["feature", "coefficient", "expected_sign", "verdict"])
print(signs.to_string(index=False))
signs.to_csv(f"{OUT}/C_coefficient_signs.csv", index=False)

# Variance Inflation Factor: how much a feature is "explained" by the other features.
# Rule of thumb: above ~5 is a warning, above ~10 is a serious concern.
Xc = sm.add_constant(X_scaled_val[is_dev], has_constant="add")
vif = pd.DataFrame(
    {"feature": spec["features"], "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(spec["features"]))]}
)
print("\n" + vif.round(2).to_string(index=False))
vif.to_csv(f"{OUT}/C_vif.csv", index=False)

# ===========================================================================
# D. DISCRIMINATION - can the model rank risky customers above safe ones?
# ===========================================================================
heading("D. DISCRIMINATION (AUC, Gini, KS)")
score = dev_pred["pd_model"].values
disc_dev = metrics(y[is_dev], score[is_dev])
disc_hold = metrics(y[is_hold], score[is_hold])
disc = pd.DataFrame({"development": disc_dev, "holdout": disc_hold}).T
print(disc.round(4).to_string())
disc.to_csv(f"{OUT}/D_discrimination.csv")
results["disc_dev"], results["disc_hold"] = disc_dev, disc_hold
results["auc_gap_dev_minus_hold"] = disc_dev["AUC"] - disc_hold["AUC"]
print(f"AUC gap (development - hold-out): {results['auc_gap_dev_minus_hold']:.4f}  (a big gap would suggest overfitting)")

# ===========================================================================
# E. CALIBRATION - are the predicted probabilities close to reality?
# ===========================================================================
heading("E. CALIBRATION (predicted PD vs actual default rate, by decile, hold-out)")
hold_df = pd.DataFrame({"pd": score[is_hold], "y": y[is_hold]})
hold_df["decile"] = pd.qcut(hold_df["pd"], 10, labels=False) + 1  # 1 = lowest risk ... 10 = highest
cal = hold_df.groupby("decile").agg(customers=("y", "size"), defaults=("y", "sum"),
                                   avg_predicted_pd=("pd", "mean"), actual_default_rate=("y", "mean"))
# Binomial test: if the predicted PD were exactly right, how surprising is the number of defaults we saw?
cal["binomial_p_value"] = [
    binomtest(int(r.defaults), int(r.customers), float(r.avg_predicted_pd)).pvalue for r in cal.itertuples()
]
print(cal.round(4).to_string())
cal.to_csv(f"{OUT}/E_calibration_deciles.csv")

overall_pred = float(hold_df["pd"].mean())
overall_act = float(hold_df["y"].mean())
brier_model = brier_score_loss(hold_df["y"], hold_df["pd"])
brier_naive = brier_score_loss(hold_df["y"], np.full(len(hold_df), y[is_dev].mean()))
print(f"\nOverall hold-out: average predicted PD {overall_pred:.4f} vs actual default rate {overall_act:.4f}")
print(f"Brier score (lower = better): model {brier_model:.4f} vs naive 'everyone is average' {brier_naive:.4f}")
results.update(overall_pred_pd=overall_pred, overall_actual_dr=overall_act,
               brier_model=brier_model, brier_naive=brier_naive,
               deciles_binomial_p_below_5pct=int((cal["binomial_p_value"] < 0.05).sum()))
print(f"Deciles where the binomial test rejects the predicted PD at 5%: {results['deciles_binomial_p_below_5pct']} of 10")

# Diagnostic: where does the calibration miss come from? Compare predicted vs actual for each
# value of the most important input (latest repayment status, PAY_1).
diag_df = pd.DataFrame({"PAY_1": raw.loc[is_hold, "PAY_1"].values, "pd": score[is_hold], "y": y[is_hold]})
by_pay1 = diag_df.groupby("PAY_1").agg(customers=("y", "size"), avg_predicted_pd=("pd", "mean"),
                                        actual_default_rate=("y", "mean"))
by_pay1["gap_pred_minus_actual"] = by_pay1["avg_predicted_pd"] - by_pay1["actual_default_rate"]
print("\nDiagnostic - predicted vs actual by latest repayment status (PAY_1), hold-out:")
print(by_pay1.round(3).to_string())
by_pay1.to_csv(f"{OUT}/E2_calibration_by_pay1.csv")

# ===========================================================================
# F. STABILITY - PSI (Population Stability Index)
# ===========================================================================
heading("F. STABILITY (PSI, development vs hold-out)")


def psi(expected, actual, bins=10):
    """PSI = sum over bins of (share_actual - share_expected) * ln(share_actual / share_expected).
    Rule of thumb: <0.10 stable, 0.10-0.25 some shift, >0.25 big shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)  # avoid log(0)
    return float(np.sum((a - e) * np.log(a / e)))


psi_rows = [["MODEL SCORE (PD)", psi(score[is_dev], score[is_hold])]]
for f in spec["features"]:
    psi_rows.append([f, psi(X_val.loc[is_dev, f].values, X_val.loc[is_hold, f].values)])
psi_df = pd.DataFrame(psi_rows, columns=["variable", "PSI"])
psi_df["status"] = np.where(psi_df["PSI"] < 0.10, "stable", np.where(psi_df["PSI"] < 0.25, "shift", "big shift"))
print(psi_df.round(4).to_string(index=False))
psi_df.to_csv(f"{OUT}/F_psi.csv", index=False)
results["psi_score"] = float(psi_df.loc[0, "PSI"])
results["psi_max_feature"] = float(psi_df["PSI"].iloc[1:].max())

# ===========================================================================
# G. BENCHMARKING - simple and complex alternatives
# ===========================================================================
heading("G. BENCHMARKING (naive, one-variable, developer model, challenger)")
challenger = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=3, early_stopping=True, random_state=SEED
)
challenger.fit(X_val[is_dev], y[is_dev])
pd_chal = challenger.predict_proba(X_val)[:, 1]

bench = {
    "Naive (everyone gets the average PD)": {"AUC": 0.5, "Gini": 0.0, "KS": 0.0,
                                             "Brier": brier_naive},
    "One variable only (latest repayment status)": {**metrics(y[is_hold], X_val.loc[is_hold, "PAY_1"].values),
                                                    "Brier": np.nan},
    "Developer model (logistic regression)": {**disc_hold, "Brier": brier_model},
    "Challenger (gradient boosting, same inputs)": {**metrics(y[is_hold], pd_chal[is_hold]),
                                                    "Brier": brier_score_loss(y[is_hold], pd_chal[is_hold])},
}
bench_df = pd.DataFrame(bench).T
print(bench_df.round(4).to_string())
bench_df.to_csv(f"{OUT}/G_benchmark.csv")
results["bench"] = {k: {m: (None if pd.isna(v) else float(v)) for m, v in row.items()} for k, row in bench_df.iterrows()}

# ===========================================================================
# H. SENSITIVITY - what if conditions worsen? (simple "what-if" shocks)
# ===========================================================================
heading("H. SENSITIVITY (what-if shocks applied to hold-out customers)")


def score_raw(df):
    """Run raw customer data through the validator's rebuilt model."""
    feats = (rebuild_features(df) - mean) / std
    return refit.predict_proba(feats)[:, 1]


hold_raw = raw[is_hold].copy()
base = score_raw(hold_raw).mean()


def shock_bills(df):
    d = df.copy(); d[[f"BILL_AMT{m}" for m in range(1, 7)]] *= 1.2; return d          # balances 20% higher


def shock_payments(df):
    d = df.copy(); d[[f"PAY_AMT{m}" for m in range(1, 7)]] *= 0.8; return d           # payments 20% lower


def shock_slip(df):
    # applies only to accounts already using credit (PAY_1 >= 0); accounts that paid in full are left alone
    d = df.copy()
    d["PAY_1"] = np.where(d["PAY_1"] >= 0, d["PAY_1"] + 1, d["PAY_1"])                 # latest month slips one notch
    return d


def shock_limit(df):
    d = df.copy(); d["LIMIT_BAL"] *= 0.8; return d                                     # credit limit cut by 20%


shocks = {
    "Balances +20%": shock_bills,
    "Payments -20%": shock_payments,
    "Latest repayment slips one notch": shock_slip,
    "Credit limit -20%": shock_limit,
    "All four combined (severe)": lambda d: shock_limit(shock_slip(shock_payments(shock_bills(d)))),
}
sens_rows = [["Baseline (no shock)", base, 0.0, 0.0]]
for name, fn in shocks.items():
    shocked = score_raw(fn(hold_raw)).mean()
    sens_rows.append([name, shocked, shocked - base, (shocked / base - 1)])
sens = pd.DataFrame(sens_rows, columns=["scenario", "avg_predicted_pd", "change_in_pd_points", "relative_change"])
print(sens.round(4).to_string(index=False))
sens.to_csv(f"{OUT}/H_sensitivity.csv", index=False)
results["sensitivity"] = sens.set_index("scenario")["avg_predicted_pd"].to_dict()
results["sensitivity_all_directions_up"] = bool((sens["change_in_pd_points"].iloc[1:] > 0).all())

# ===========================================================================
# Charts
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.5, 5))
for label, mask, s in [("Developer - development", is_dev, score), ("Developer - hold-out", is_hold, score),
                       ("Challenger - hold-out", is_hold, pd_chal)]:
    fpr, tpr, _ = roc_curve(y[mask], s[mask])
    ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y[mask], s[mask]):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves")
ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/roc_curves.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(cal["avg_predicted_pd"], cal["actual_default_rate"], "o-", label="Model (by decile)")
top = max(cal["avg_predicted_pd"].max(), cal["actual_default_rate"].max()) * 1.05
ax.plot([0, top], [0, top], "k--", label="Perfect calibration")
ax.set_xlabel("Average predicted PD"); ax.set_ylabel("Actual default rate")
ax.set_title("Calibration on hold-out sample"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/calibration_plot.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4))
plot_df = sens.iloc[1:]
colours = ["#c0504d" if v > 0 else "#7f7f7f" for v in plot_df["change_in_pd_points"]]  # grey = PD went DOWN (unexpected)
ax.barh(plot_df["scenario"], plot_df["change_in_pd_points"] * 100, color=colours)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Change in average predicted PD (percentage points)")
ax.set_title("Sensitivity of predicted PD to what-if shocks (grey = PD fell)")
ax.invert_yaxis(); fig.tight_layout(); fig.savefig(f"{OUT}/sensitivity_plot.png", dpi=150); plt.close(fig)

json.dump(results, open(f"{OUT}/key_results.json", "w"), indent=2, default=float)
print("\nAll validation outputs saved to", OUT)
