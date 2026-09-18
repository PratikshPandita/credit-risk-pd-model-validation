# Independent Validation Report — Credit-Card PD Model

**Model validated:** Logistic-regression Probability-of-Default (PD) model (`01_develop_pd_model.py`)
**Validator materials:** raw data, developer's model documentation (`outputs/developer_model_spec.json`), developer's predictions
**Tests run by:** `02_validate_pd_model.py` (results in `outputs/validation/`)
**Data:** 24,000 credit-card accounts, 22.11% defaulted (missed next month's payment). Split 70% development (16,800) / 30% hold-out (7,200).

> **Independence disclosure.** This is a portfolio project. The developer and validator roles were both played by the same person, so it demonstrates the *method* of independent validation, not true independence. In a bank these are separate teams.

---

## 1. Overall conclusion

The model can be **reproduced exactly from its documentation**, ranks customers well (Gini 0.49) and shows **no sign of overfitting** (development and hold-out results are almost identical). On average its predicted PD matches the actual default rate (22.0% vs 22.1%).

However, it has a **wrong-signed input that makes the model behave illogically under stress**, and its probabilities are **mis-calibrated for several customer groups**. My assessment:

- **Ranking customers by risk:** acceptable.
- **Use in stress testing / scenario analysis:** **not yet fit for use** until Findings 1 and 2 are fixed.

## 2. Summary of findings

| ID | Severity | Area | Finding | Recommendation | Status |
|----|----------|------|---------|----------------|--------|
| 1 | **High** | Conceptual soundness | `PAYMENT_RATIO` has the wrong sign: the more of the bill a customer pays, the *higher* the predicted PD (coefficient +0.18; expected negative). Consequence: in the sensitivity test, 20% lower payments and 20% higher balances make average PD *fall* (−0.31 and −0.18 percentage points). | Investigate how the ratio is built (e.g. customers with no bill are set to 1.0), test alternative definitions or remove it, then re-run the sensitivity tests. | Open |
| 2 | **Medium** | Calibration | Predicted PDs are off for 4 of 10 risk deciles (binomial test, 5% level). The safest decile is under-predicted (7.3% predicted vs 11.1% actual). By repayment status, customers 2 months late are under-predicted by ~19 points (54.6% vs 73.3%) while `PAY_1 = 0` and `1` are over-predicted. Likely cause: `PAY_1` is fed into the model as a straight-line number although its codes (−2, −1, 0, 1, 2 …) do not have a straight-line relationship with risk. | Treat `PAY_1` as categories (e.g. −2/−1, 0, 1, 2, 3+) and re-test calibration. | Open |
| 3 | **Medium** | Benchmarking | A gradient-boosting challenger on the same inputs is clearly better (AUC 0.781 vs 0.743; Gini 0.561 vs 0.486). The logistic model leaves performance unused, probably through the non-linearity in Finding 2. | Fix Finding 2 and re-compare. If a gap remains, document why the simpler model is preferred (e.g. interpretability). | Open |
| 4 | **Medium** | Testing limitations | The data has no dates, so an out-of-time back-test is impossible. Stability (PSI) on a random split is near zero by construction, so it gives little real assurance. | Require an out-of-time test and a monitoring plan once dated data exists. | Open |
| 5 | Low | Conceptual soundness | Two inputs add little: `AVG_UTILISATION` (p = 0.45) and `EDU_GRADUATE` (p = 0.39) are not statistically significant. | Consider removing them for a simpler model. | Open |
| 6 | Low | Data / documentation | Codes outside the documented values: `EDUCATION` 0/5/6 (267 rows), `MARRIAGE` 0 (42 rows). `PAY_1` values −2 and 0 are not in the original data dictionary yet cover 58% of accounts. 24 exact duplicate rows. | Document the meaning assumed for each code and how they were treated. | Open |
| 7 | Low | Fair lending | `AGE` is used and has a positive effect (older → higher PD). `SEX` and `MARRIAGE` were correctly excluded. | Refer the use of `AGE` to a fair-lending / compliance review. | Open |

## 3. Test results

### A. Data quality
24,000 rows, no missing values, 24 exact duplicates, 1,539 accounts with a negative bill (credit balance). See Finding 6 for undocumented codes.

### B. Replication — passed
Rebuilding the features from the documentation and applying the developer's coefficients reproduces the developer's PDs (maximum difference 3×10⁻¹⁶). Independently re-fitting the model with a different library (scikit-learn instead of statsmodels) gives PDs within 1.3×10⁻⁷ and coefficients within 1.1×10⁻⁷.

### C. Conceptual soundness
All signs match business expectation except `PAYMENT_RATIO` (Finding 1). Multicollinearity is acceptable: the highest VIF is 3.6 (`MAX_DELINQ_6M`); values above 5 would be a concern.

### D. Discrimination (does it rank risky customers above safe ones?) — good

| Sample | AUC | Gini | KS |
|---|---|---|---|
| Development | 0.745 | 0.490 | 0.403 |
| Hold-out | 0.743 | 0.486 | 0.412 |

The AUC gap between the two samples is 0.002, so there is no evidence of overfitting. Chart: `outputs/validation/roc_curves.png`.

### E. Calibration (are the predicted probabilities right?) — mixed
Overall the model is right on average (predicted 22.01% vs actual 22.11%) and beats the naive "everyone is average" forecast on Brier score (0.141 vs 0.172). By decile:

| Decile | Avg predicted PD | Actual default rate | Binomial p-value |
|---|---|---|---|
| 1 (safest) | 7.3% | 11.1% | 0.0002 |
| 2 | 10.4% | 11.4% | 0.39 |
| 3 | 11.7% | 9.7% | 0.10 |
| 4 | 12.8% | 9.3% | 0.004 |
| 5 | 14.1% | 13.1% | 0.49 |
| 6 | 15.8% | 15.0% | 0.61 |
| 7 | 18.6% | 15.7% | 0.044 |
| 8 | 26.3% | 25.7% | 0.74 |
| 9 | 37.6% | 42.9% | 0.003 |
| 10 (riskiest) | 65.5% | 67.2% | 0.35 |

Chart: `outputs/validation/calibration_plot.png`. Breakdown by repayment status: `outputs/validation/E2_calibration_by_pay1.csv`.

### F. Stability (PSI, development vs hold-out) — stable, but weak evidence
Score PSI 0.002 and the largest input PSI 0.005, far below the 0.10 "some shift" threshold. Because the two samples are random halves of the same data, this is expected (Finding 4).

### G. Benchmarking (hold-out)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Naive (everyone average) | 0.500 | 0.000 | 0.000 |
| One input only (latest repayment status) | 0.685 | 0.369 | 0.365 |
| **Developer model** | **0.743** | **0.486** | **0.412** |
| Challenger (gradient boosting) | 0.781 | 0.561 | 0.443 |

### H. Sensitivity (what-if shocks on hold-out customers; illustrative, not macro-calibrated)

| Scenario | Avg predicted PD | Change (points) |
|---|---|---|
| Baseline | 22.0% | — |
| Balances +20% | 21.8% | −0.2 |
| Payments −20% | 21.7% | −0.3 |
| Latest repayment slips one notch | 29.6% | +7.6 |
| Credit limit −20% | 22.8% | +0.8 |
| All four combined | 30.0% | +8.0 |

The first two rows move in the **wrong direction** (Finding 1). Chart: `outputs/validation/sensitivity_plot.png`.

## 4. Scope limits (what a full bank validation would add)

- The data is a 2005 Taiwan credit-card snapshot with a 1-month default definition, not a US portfolio with 12-month PDs. LGD and EAD models are not covered.
- No macroeconomic variables, so scenario shocks are simple what-ifs, not regulatory (CCAR-style) scenarios.
- Not reviewed: governance and approval process, documentation against internal policy, production implementation, ongoing-monitoring procedures.
