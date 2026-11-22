"""
ITDMA3-22 Group 1 -- Standardized Model Suite
Fixes applied vs. the three earlier, separately built versions:
  1. One common feature set and train/test split across OLS, Ridge,
     Lasso and Random Forest (each earlier version used a
     different spec -- not comparable as-is)
  2. Ridge/Lasso alpha chosen by proper time-series cross-validation
     (an earlier Ridge/Lasso had collapsed to predicting
     the training mean -- alpha was never tuned)
  3. mlr_avg_mw used alone as the loadshedding measure, not alongside
     ls_hours (that combination caused severe multicollinearity in
     an earlier Linear Regression)
  4. Adds the dv_bci track, so "business performance" is modelled
     both as port throughput AND as business confidence, not ports only
Each of OLS/Ridge/Lasso/RF is run once per outcome (dv_portwatch_tonnes,
dv_bci) -- 8 models total, one shared comparison table.
"""

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
FIGURES = "figures"
import os; os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
# diesel_level dropped: r=0.75 with petrol95_level -> severe multicollinearity
# for the linear models (this is what wrecked the plain Linear Regression in
# an earlier version that included both). petrol95_level alone represents
# fuel price here; diesel_level is reported separately as a robustness check.
# Note: mlr_avg_mw and cpi_yoy_pct are also correlated at r=0.76 (loadshedding
# severity tracks the 2022-2023 inflation spike) -- kept in since both are
# substantively necessary, but flagged via VIF in the diagnostics output.
FEATURES = ["mlr_avg_mw", "petrol95_level",
            "cpi_yoy_pct", "unemployment_rate_pct", "usdzar_avg"]
N_TEST = 9
RANDOM_STATE = 42

df = pd.read_csv("data/processed/group1_monthly_merged.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month").reset_index(drop=True)

all_metrics = []
all_coefs = {}
hyperparam_log = []
ols_summaries = []


def run_outcome(dv):
    model_df = df.dropna(subset=[dv] + FEATURES).reset_index(drop=True)

    train = model_df.iloc[:-N_TEST].copy()
    test = model_df.iloc[-N_TEST:].copy()

    # Standardise on TRAIN stats only (no leakage)
    means, stds = train[FEATURES].mean(), train[FEATURES].std()
    train_s, test_s = train.copy(), test.copy()
    train_s[FEATURES] = (train[FEATURES] - means) / stds
    test_s[FEATURES] = (test[FEATURES] - means) / stds

    # Interaction computed on the already-standardized (mean-centered)
    # features, not raw scale -- this is what keeps it from being almost
    # perfectly collinear with the main effects it's built from.
    train_s["mlr_x_petrol95"] = train_s["mlr_avg_mw"] * train_s["petrol95_level"]
    test_s["mlr_x_petrol95"] = test_s["mlr_avg_mw"] * test_s["petrol95_level"]
    feats = FEATURES + ["mlr_x_petrol95"]

    y_train, y_test = train[dv].values, test[dv].values
    naive_rmse = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))

    results = {}

    # --- OLS (HAC-robust SEs) ---
    X_train = sm.add_constant(train_s[feats])
    X_test = sm.add_constant(test_s[feats], has_constant="add")
    ols = sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    pred = ols.predict(X_test)
    results["OLS"] = (pred, ols.params[1:].values)
    all_coefs[f"OLS_{dv}"] = dict(zip(feats, ols.params[1:].values))

    # Full OLS summary (coefficient, HAC-robust SE, p-value, 95% CI) --
    # this is what the Model Evaluation section's significance claims
    # (e.g. "interaction term p<0.001") are verified against. Previously
    # only point estimates were saved, not the p-values themselves.
    ci = ols.conf_int()
    ci = ci.values if hasattr(ci, "values") else ci
    for name, coef, se, pval, cilo, cihi in zip(
        ["const"] + feats, ols.params, ols.bse, ols.pvalues, ci[:, 0], ci[:, 1]):
        ols_summaries.append({"outcome": dv, "feature": name, "coef": coef, "std_err": se,
                               "p_value": pval, "ci_lower": cilo, "ci_upper": cihi,
                               "significant_at_0.05": pval < 0.05})

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.Series(
        [variance_inflation_factor(X_train.values, i) for i in range(1, X_train.shape[1])],
        index=feats, name="VIF")
    vif.to_csv(f"{RESULTS}/vif_{dv}.csv")

    # --- Ridge (alpha chosen via TimeSeriesSplit CV, not fixed) ---
    tscv = TimeSeriesSplit(n_splits=3)
    ridge_alphas = np.logspace(-2, 3, 50)
    ridge = RidgeCV(alphas=ridge_alphas, cv=tscv).fit(train_s[feats], y_train)
    pred = ridge.predict(test_s[feats])
    results["Ridge"] = (pred, ridge.coef_)
    all_coefs[f"Ridge_{dv}"] = dict(zip(feats, ridge.coef_))
    hyperparam_log.append({"outcome": dv, "model": "Ridge", "alpha": ridge.alpha_,
                            "search_max": ridge_alphas.max(), "hit_ceiling": ridge.alpha_ == ridge_alphas.max(),
                            "nonzero_coefs": len(feats), "n_features": len(feats)})

    # --- Lasso (alpha chosen via TimeSeriesSplit CV) ---
    # NOTE: an earlier version searched only up to alpha=1000 and the CV
    # selection hit that ceiling for dv_portwatch_tonnes -- meaning the true
    # CV-optimal alpha was never actually found, only an artificially capped
    # one. Widened to 1e6 so CV can find its real minimum, whatever that is.
    lasso_alphas = np.logspace(-2, 6, 100)
    lasso = LassoCV(alphas=lasso_alphas, cv=tscv, max_iter=50000).fit(train_s[feats], y_train)
    pred = lasso.predict(test_s[feats])
    results["Lasso"] = (pred, lasso.coef_)
    all_coefs[f"Lasso_{dv}"] = dict(zip(feats, lasso.coef_))
    hyperparam_log.append({"outcome": dv, "model": "Lasso", "alpha": lasso.alpha_,
                            "search_max": lasso_alphas.max(), "hit_ceiling": lasso.alpha_ == lasso_alphas.max(),
                            "nonzero_coefs": int((lasso.coef_ != 0).sum()), "n_features": len(feats)})

    # --- Random Forest (small grid, time-series CV, time-series CV) ---
    param_grid = {"max_depth": [2, 3, None], "min_samples_leaf": [2, 4], "max_features": [1.0, "sqrt"]}
    rf = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
                       param_grid, cv=tscv, scoring="neg_mean_absolute_error")
    rf.fit(train_s[feats], y_train)
    pred = rf.predict(test_s[feats])
    results["Random Forest"] = (pred, rf.best_estimator_.feature_importances_)
    all_coefs[f"RF_{dv}"] = dict(zip(feats, rf.best_estimator_.feature_importances_))

    for name, (pred, _) in results.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        all_metrics.append({"outcome": dv, "model": name, "n_train": len(train), "n_test": len(test),
                             "MAE": mae, "RMSE": rmse, "R2": r2, "naive_RMSE": naive_rmse})

    # Actual vs predicted plot
    plt.figure(figsize=(7, 4))
    plt.plot(test["month"], y_test, "o-", label="Actual", color="black")
    for name, (pred, _) in results.items():
        plt.plot(test["month"], pred, "--", marker="x", label=name)
    plt.title(f"Actual vs Predicted — {dv} (held-out test months)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/actual_vs_predicted_{dv}.png", dpi=120)
    plt.close()

    return model_df, train, test


for dv in ["dv_portwatch_tonnes", "dv_bci"]:
    run_outcome(dv)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(f"{RESULTS}/model_comparison_table.csv", index=False)
with open(f"{RESULTS}/coefficients_and_importances.json", "w") as f:
    json.dump(all_coefs, f, indent=2)

pd.DataFrame(hyperparam_log).to_csv(f"{RESULTS}/selected_hyperparameters.csv", index=False)
pd.DataFrame(ols_summaries).to_csv(f"{RESULTS}/ols_pvalues.csv", index=False)
print(f"Saved: {RESULTS}/selected_hyperparameters.csv, {RESULTS}/ols_pvalues.csv")

# Coefficient comparison (OLS / Ridge / Lasso), standardized scale -- per DV
feat_labels = FEATURES + ["mlr_x_petrol95"]
for dv in ["dv_portwatch_tonnes", "dv_bci"]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(len(feat_labels))
    for i, model in enumerate(["OLS", "Ridge", "Lasso"]):
        vals = [all_coefs[f"{model}_{dv}"][f] for f in feat_labels]
        ax.bar(x + (i - 1) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Standardized coefficients — {dv}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/coefficients_{dv}.png", dpi=120)
    plt.close()

    # RF feature importance
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [all_coefs[f"RF_{dv}"][f] for f in feat_labels]
    ax.barh(feat_labels, vals, color="#2a7")
    ax.set_title(f"Random Forest feature importance — {dv}")
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/rf_importance_{dv}.png", dpi=120)
    plt.close()

print(metrics_df.to_string(index=False))
print(f"\nSaved: {RESULTS}/model_comparison_table.csv, {RESULTS}/coefficients_and_importances.json")
