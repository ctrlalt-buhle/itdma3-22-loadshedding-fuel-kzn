"""
ITDMA3-22 Group 1 -- Evaluation Suite
Builds on standardized_model_suite.py (same data, same feature spec,
same tuning) and adds the evaluation evidence Deliverable 5 asks for:

  TC1  Baseline hold-out (last 9 months), incl. train-vs-test gap
  TC2  Rolling-origin (expanding window) evaluation, 3-month blocks
  TC3  Test-window sensitivity (6 / 9 / 12 month hold-outs)
  TC4  Outlier / influence sensitivity (Cook's distance)
  TC5  Feature-specification sensitivity
  TC6  Random Forest seed stability (10 seeds)
  TC7  Moving-block bootstrap of the interaction coefficient
  TC8  Cross-outcome replication (port throughput vs BCI)
  +    OLS residual diagnostics, permutation importance, interaction plots

Run from the repository root:  python3 src/evaluation_suite.py
Everything is deterministic (fixed seeds); outputs go to results/eval_*.csv
and figures/eval_*.png.
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RES, FIG = "results", "figures"
import os; os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 42
BASE = ["mlr_avg_mw", "petrol95_level", "cpi_yoy_pct", "unemployment_rate_pct", "usdzar_avg"]
DVS = {"dv_portwatch_tonnes": "Port throughput", "dv_bci": "Business confidence"}
MODELS = ["OLS", "Ridge", "Lasso", "Random Forest"]

df = pd.read_csv("data/processed/group1_monthly_merged.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month").reset_index(drop=True)
df["mlr_lag1"] = df["mlr_avg_mw"].shift(1)


def build(dv, feats, n_test, fuel="petrol95_level", interaction=True, frame=None):
    """Time-ordered split, train-only standardisation, interaction on z-scores."""
    d = (frame if frame is not None else df).dropna(subset=[dv] + feats).reset_index(drop=True)
    train, test = d.iloc[:-n_test].copy(), d.iloc[-n_test:].copy()
    mu, sd = train[feats].mean(), train[feats].std()
    tr, te = (train[feats] - mu) / sd, (test[feats] - mu) / sd
    cols = list(feats)
    if interaction:
        tr["mlr_x_fuel"] = tr["mlr_avg_mw"] * tr[fuel]
        te["mlr_x_fuel"] = te["mlr_avg_mw"] * te[fuel]
        cols.append("mlr_x_fuel")
    return dict(train=train, test=test, Xtr=tr[cols], Xte=te[cols], cols=cols,
                ytr=train[dv].values, yte=test[dv].values, mu=mu, sd=sd)


def fit_all(S, seed=SEED, fast=False):
    """Fit OLS / Ridge / Lasso / RF exactly as in the Deliverable 4 suite."""
    tscv = TimeSeriesSplit(n_splits=3)
    out, objs = {}, {}
    ols = sm.OLS(S["ytr"], sm.add_constant(S["Xtr"])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    out["OLS"] = ols.predict(sm.add_constant(S["Xte"], has_constant="add"))
    objs["OLS"] = ols
    r = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=tscv).fit(S["Xtr"], S["ytr"])
    out["Ridge"] = r.predict(S["Xte"]); objs["Ridge"] = r
    l = LassoCV(alphas=np.logspace(-2, 6, 100), cv=tscv, max_iter=50000).fit(S["Xtr"], S["ytr"])
    out["Lasso"] = l.predict(S["Xte"]); objs["Lasso"] = l
    grid = {"max_depth": [2, 3, None], "min_samples_leaf": [2, 4], "max_features": [1.0, "sqrt"]}
    rf = GridSearchCV(RandomForestRegressor(n_estimators=200 if fast else 300, random_state=seed),
                      grid, cv=tscv, scoring="neg_mean_absolute_error").fit(S["Xtr"], S["ytr"])
    out["Random Forest"] = rf.predict(S["Xte"]); objs["Random Forest"] = rf.best_estimator_
    return out, objs


def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))


# ---------------------------------------------------------------- TC1
tc1, store = [], {}
for dv in DVS:
    S = build(dv, BASE, 9)
    preds, objs = fit_all(S)
    store[dv] = (S, preds, objs)
    naive = rmse(S["yte"], np.full(9, S["ytr"].mean()))
    persist = rmse(S["yte"], np.full(9, S["ytr"][-1]))
    for m in MODELS:
        tr_pred = (objs[m].predict(sm.add_constant(S["Xtr"])) if m == "OLS" else objs[m].predict(S["Xtr"]))
        tc1.append(dict(outcome=dv, model=m,
                        train_RMSE=rmse(S["ytr"], tr_pred), train_R2=r2_score(S["ytr"], tr_pred),
                        test_MAE=mean_absolute_error(S["yte"], preds[m]), test_RMSE=rmse(S["yte"], preds[m]),
                        test_R2=r2_score(S["yte"], preds[m]), naive_mean_RMSE=naive, naive_last_value_RMSE=persist,
                        skill_vs_mean=1 - rmse(S["yte"], preds[m]) / naive,
                        overfit_ratio=rmse(S["yte"], preds[m]) / rmse(S["ytr"], tr_pred)))
tc1 = pd.DataFrame(tc1); tc1.to_csv(f"{RES}/eval_tc1_holdout.csv", index=False)
print("TC1 done")

# ---------------------------------------------------------------- TC2 rolling origin
tc2_rows, tc2_pred = [], []
for dv in DVS:
    d = df.dropna(subset=[dv] + BASE).reset_index(drop=True)
    for t0 in range(24, len(d) - 2, 3):
        sub = d.iloc[:t0 + 3].copy()
        S = build(dv, BASE, 3, frame=sub)
        preds, _ = fit_all(S, fast=True)
        for m in MODELS:
            for mo, a, p in zip(S["test"]["month"], S["yte"], preds[m]):
                tc2_pred.append(dict(outcome=dv, origin=t0, model=m, month=mo, actual=a, pred=p))
        for nm, val in [("Naive mean", S["ytr"].mean()), ("Naive last value", S["ytr"][-1])]:
            for mo, a in zip(S["test"]["month"], S["yte"]):
                tc2_pred.append(dict(outcome=dv, origin=t0, model=nm, month=mo, actual=a, pred=val))
tc2p = pd.DataFrame(tc2_pred); tc2p.to_csv(f"{RES}/eval_tc2_rolling_predictions.csv", index=False)
tc2 = (tc2p.assign(se=(tc2p.actual - tc2p.pred) ** 2, ae=(tc2p.actual - tc2p.pred).abs())
       .groupby(["outcome", "model"]).agg(RMSE=("se", lambda x: float(np.sqrt(x.mean()))), MAE=("ae", "mean"),
                                           n_forecasts=("se", "size"), n_origins=("origin", "nunique")).reset_index())
nm = tc2[tc2.model == "Naive mean"].set_index("outcome")["RMSE"]
tc2["skill_vs_naive_mean"] = tc2.apply(lambda r: 1 - r.RMSE / nm[r.outcome], axis=1)
tc2.to_csv(f"{RES}/eval_tc2_rolling_origin.csv", index=False)
print("TC2 done")

# ---------------------------------------------------------------- TC3 test-window sensitivity
tc3 = []
for dv in DVS:
    for nt in (6, 9, 12):
        S = build(dv, BASE, nt)
        preds, _ = fit_all(S, fast=True)
        naive = rmse(S["yte"], np.full(nt, S["ytr"].mean()))
        for m in MODELS:
            tc3.append(dict(outcome=dv, n_test=nt, n_train=len(S["ytr"]), model=m,
                            RMSE=rmse(S["yte"], preds[m]), naive_RMSE=naive,
                            ratio_to_naive=rmse(S["yte"], preds[m]) / naive))
pd.DataFrame(tc3).to_csv(f"{RES}/eval_tc3_window_sensitivity.csv", index=False)
print("TC3 done")

# ---------------------------------------------------------------- TC4 influence / outliers
tc4 = []
for dv in DVS:
    S = store[dv][0]
    X = sm.add_constant(S["Xtr"])
    plain = sm.OLS(S["ytr"], X).fit()
    cooks = plain.get_influence().cooks_distance[0]
    thr = 4 / len(S["ytr"])
    top = np.argsort(cooks)[::-1][:3]
    infl = S["train"]["month"].iloc[top].dt.strftime("%Y-%m").tolist()
    keep = np.ones(len(S["ytr"]), bool); keep[top] = False
    for label, mask in [("All 34 training months", np.ones(len(keep), bool)), ("Drop 3 most influential", keep)]:
        f = sm.OLS(S["ytr"][mask], X[mask]).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        tc4.append(dict(outcome=dv, sample=label, n=int(mask.sum()), interaction_coef=f.params["mlr_x_fuel"],
                        interaction_p=f.pvalues["mlr_x_fuel"], r2_in_sample=f.rsquared,
                        influential_months=", ".join(infl) if label != "All 34 training months" else "",
                        n_above_4_over_n=int((cooks > thr).sum())))
pd.DataFrame(tc4).to_csv(f"{RES}/eval_tc4_influence.csv", index=False)
print("TC4 done")

# ---------------------------------------------------------------- TC5 specification sensitivity
tc5 = []
specs = {
    "Base (5 features + interaction)": dict(feats=BASE, fuel="petrol95_level", interaction=True),
    "Diesel instead of petrol": dict(feats=["mlr_avg_mw", "diesel_level", "cpi_yoy_pct", "unemployment_rate_pct", "usdzar_avg"], fuel="diesel_level", interaction=True),
    "Drop CPI (collinear with loadshedding)": dict(feats=[f for f in BASE if f != "cpi_yoy_pct"], fuel="petrol95_level", interaction=True),
    "Add 1-month lag of loadshedding": dict(feats=BASE + ["mlr_lag1"], fuel="petrol95_level", interaction=True),
    "No interaction term": dict(feats=BASE, fuel="petrol95_level", interaction=False),
}
for dv in DVS:
    for name, sp in specs.items():
        S = build(dv, sp["feats"], 9, fuel=sp["fuel"], interaction=sp["interaction"])
        preds, objs = fit_all(S, fast=True)
        o = objs["OLS"]
        row = dict(outcome=dv, spec=name, n_train=len(S["ytr"]),
                   interaction_coef=o.params.get("mlr_x_fuel", np.nan), interaction_p=o.pvalues.get("mlr_x_fuel", np.nan),
                   ci_low=o.conf_int().loc["mlr_x_fuel", 0] if "mlr_x_fuel" in o.params else np.nan,
                   ci_high=o.conf_int().loc["mlr_x_fuel", 1] if "mlr_x_fuel" in o.params else np.nan,
                   naive_RMSE=rmse(S["yte"], np.full(9, S["ytr"].mean())))
        for m in MODELS: row[f"{m}_test_RMSE"] = rmse(S["yte"], preds[m])
        tc5.append(row)
pd.DataFrame(tc5).to_csv(f"{RES}/eval_tc5_spec_sensitivity.csv", index=False)
print("TC5 done")

# ---------------------------------------------------------------- TC6 RF seed stability
tc6 = []
for dv in DVS:
    S = store[dv][0]
    vals = []
    for sd in range(10):
        rf = GridSearchCV(RandomForestRegressor(n_estimators=200, random_state=sd),
                          {"max_depth": [2, 3, None], "min_samples_leaf": [2, 4], "max_features": [1.0, "sqrt"]},
                          cv=TimeSeriesSplit(3), scoring="neg_mean_absolute_error").fit(S["Xtr"], S["ytr"])
        vals.append(rmse(S["yte"], rf.predict(S["Xte"])))
    tc6.append(dict(outcome=dv, seeds=10, RMSE_mean=np.mean(vals), RMSE_std=np.std(vals),
                    RMSE_min=min(vals), RMSE_max=max(vals)))
pd.DataFrame(tc6).to_csv(f"{RES}/eval_tc6_rf_seed_stability.csv", index=False)
print("TC6 done")

# ---------------------------------------------------------------- TC7 moving-block bootstrap
rng = np.random.default_rng(SEED)
tc7, boots = [], {}
for dv in DVS:
    S = store[dv][0]
    X = sm.add_constant(S["Xtr"]).values; y = S["ytr"]; n = len(y); j = list(S["Xtr"].columns).index("mlr_x_fuel") + 1
    coefs = []
    for _ in range(1000):
        starts = rng.integers(0, n - 2, size=int(np.ceil(n / 3)))
        idx = np.concatenate([np.arange(s, s + 3) for s in starts])[:n]
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        coefs.append(b[j])
    coefs = np.array(coefs); boots[dv] = coefs
    tc7.append(dict(outcome=dv, n_boot=1000, point_estimate=float(np.linalg.lstsq(X, y, rcond=None)[0][j]),
                    boot_ci_low=np.percentile(coefs, 2.5), boot_ci_high=np.percentile(coefs, 97.5),
                    share_positive=float((coefs > 0).mean())))
pd.DataFrame(tc7).to_csv(f"{RES}/eval_tc7_block_bootstrap.csv", index=False)
print("TC7 done")

# ---------------------------------------------------------------- TC8 + residual diagnostics
tc8, diag = [], []
for dv in DVS:
    S, preds, objs = store[dv]
    o = objs["OLS"]
    res = o.resid
    lb = acorr_ljungbox(res, lags=[6], return_df=True)["lb_pvalue"].iloc[0]
    bp = het_breuschpagan(res, sm.add_constant(S["Xtr"]))[1]
    jb = jarque_bera(res)[1]
    diag.append(dict(outcome=dv, durbin_watson=durbin_watson(res), ljung_box_p_lag6=lb,
                     breusch_pagan_p=bp, jarque_bera_p=jb, resid_mean=float(np.mean(res)), resid_sd=float(np.std(res))))
    for f in S["Xtr"].columns:
        tc8.append(dict(outcome=dv, feature=f, coef=o.params[f], p_value=o.pvalues[f],
                        ci_low=o.conf_int().loc[f, 0], ci_high=o.conf_int().loc[f, 1]))
pd.DataFrame(diag).to_csv(f"{RES}/eval_residual_diagnostics.csv", index=False)
pd.DataFrame(tc8).to_csv(f"{RES}/eval_tc8_ols_by_outcome.csv", index=False)
print("TC8 + diagnostics done")

# ---------------------------------------------------------------- permutation importance
perm = []
for dv in DVS:
    S, preds, objs = store[dv]
    for m in ("Ridge", "Random Forest"):
        mod = objs[m]
        for split, X, y in (("test", S["Xte"], S["yte"]), ("train", S["Xtr"], S["ytr"])):
            r = permutation_importance(mod, X, y, n_repeats=50, random_state=SEED,
                                       scoring="neg_root_mean_squared_error")
            for f, mu_, sd_ in zip(S["Xtr"].columns, r.importances_mean, r.importances_std):
                perm.append(dict(outcome=dv, model=m, split=split, feature=f, rmse_increase=mu_, sd=sd_))
pd.DataFrame(perm).to_csv(f"{RES}/eval_permutation_importance.csv", index=False)
print("Permutation importance done")

# ================================================================ FIGURES
nice = {"mlr_avg_mw": "Loadshedding (MLR)", "petrol95_level": "Petrol price", "cpi_yoy_pct": "CPI inflation",
        "unemployment_rate_pct": "Unemployment", "usdzar_avg": "USD/ZAR", "mlr_x_fuel": "Loadshedding x Petrol"}

# 1. residual panels
for dv, lab in DVS.items():
    S, preds, objs = store[dv]
    o = objs["OLS"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(S["train"]["month"], o.resid, "o-", ms=3); ax[0, 0].axhline(0, color="k", lw=.8)
    ax[0, 0].set_title("OLS training residuals over time"); ax[0, 0].tick_params(axis="x", rotation=40)
    ax[0, 1].scatter(o.fittedvalues, o.resid, s=18); ax[0, 1].axhline(0, color="k", lw=.8)
    ax[0, 1].set_title("Residuals vs fitted (OLS, training)"); ax[0, 1].set_xlabel("Fitted")
    ax[1, 0].hist(o.resid, bins=10, color="#6a8caf"); ax[1, 0].set_title("Residual distribution (OLS, training)")
    w = 0.2; x = np.arange(len(S["yte"]))
    for i, m in enumerate(MODELS):
        ax[1, 1].bar(x + (i - 1.5) * w, S["yte"] - preds[m], w, label=m)
    ax[1, 1].axhline(0, color="k", lw=.8); ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(S["test"]["month"].dt.strftime("%b %y"), rotation=40)
    ax[1, 1].set_title("Test-month errors (actual - predicted)"); ax[1, 1].legend(fontsize=7)
    fig.suptitle(f"Residual and error analysis - {lab}"); fig.tight_layout()
    fig.savefig(f"{FIG}/eval_residuals_{dv}.png", dpi=120); plt.close(fig)

# 2. forest plot of OLS coefficients
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
t8 = pd.DataFrame(tc8)
for a, (dv, lab) in zip(axs, DVS.items()):
    d = t8[t8.outcome == dv].iloc[::-1]
    a.errorbar(d.coef, range(len(d)), xerr=[d.coef - d.ci_low, d.ci_high - d.coef], fmt="o", capsize=3)
    a.axvline(0, color="k", lw=.8); a.set_yticks(range(len(d)))
    a.set_yticklabels([nice[f] + ("  *" if p < 0.05 else "") for f, p in zip(d.feature, d.p_value)])
    a.set_title(f"{lab}: OLS coefficient (per 1 SD), 95% HAC CI"); a.set_xlabel("* = p < 0.05")
fig.tight_layout(); fig.savefig(f"{FIG}/eval_forest_ols.png", dpi=120); plt.close(fig)

# 3. interaction effect (stakeholder view, real units)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for a, (dv, lab) in zip(axs, DVS.items()):
    S, _, objs = store[dv]
    o = objs["OLS"]
    mu, sd = S["mu"], S["sd"]
    grid = np.linspace(-1, 2.2, 40)
    for z, c, nm_ in [(-1, "#2a9d8f", "Low petrol (-1 SD)"), (0, "#888", "Average petrol"), (1, "#c0392b", "High petrol (+1 SD)")]:
        row = pd.DataFrame(0.0, index=range(len(grid)), columns=S["Xtr"].columns)
        row["mlr_avg_mw"] = grid; row["petrol95_level"] = z; row["mlr_x_fuel"] = grid * z
        pred = o.predict(sm.add_constant(row, has_constant="add"))
        a.plot(grid * sd["mlr_avg_mw"] + mu["mlr_avg_mw"], pred, color=c, label=nm_)
    a.set_xlabel("Loadshedding: monthly avg manual load reduction (MW)"); a.set_ylabel(lab)
    a.set_title(f"{lab}: effect of loadshedding depends on fuel price"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/eval_interaction_effect.png", dpi=120); plt.close(fig)

# 4. permutation importance
pm = pd.DataFrame(perm)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for a, (dv, lab) in zip(axs, DVS.items()):
    d = pm[(pm.outcome == dv) & (pm.model == "Random Forest")]
    piv = d.pivot(index="feature", columns="split", values="rmse_increase").loc[list(nice)]
    piv.index = [nice[i] for i in piv.index]
    piv.plot.barh(ax=a, color=["#c0392b", "#2a9d8f"]); a.axvline(0, color="k", lw=.8)
    a.set_title(f"{lab}: Random Forest permutation importance"); a.set_xlabel("Increase in RMSE when feature is shuffled")
fig.tight_layout(); fig.savefig(f"{FIG}/eval_permutation_importance.png", dpi=120); plt.close(fig)

# 5. skill vs naive (rolling origin)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for a, (dv, lab) in zip(axs, DVS.items()):
    d = tc2[tc2.outcome == dv].sort_values("RMSE")
    a.barh(d.model, d.RMSE, color=["#999" if "Naive" in m else "#3b6ea5" for m in d.model]); a.invert_yaxis()
    a.set_title(f"{lab}: rolling-origin RMSE (lower is better)"); a.set_xlabel("RMSE")
fig.tight_layout(); fig.savefig(f"{FIG}/eval_rolling_origin_rmse.png", dpi=120); plt.close(fig)

# 6. specification sensitivity of the interaction term
t5 = pd.DataFrame(tc5); t5 = t5[t5.interaction_coef.notna()]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for a, (dv, lab) in zip(axs, DVS.items()):
    d = t5[t5.outcome == dv].iloc[::-1]
    a.errorbar(d.interaction_coef, range(len(d)), xerr=[d.interaction_coef - d.ci_low, d.ci_high - d.interaction_coef], fmt="o", capsize=3)
    a.axvline(0, color="k", lw=.8); a.set_yticks(range(len(d))); a.set_yticklabels(d.spec, fontsize=8)
    a.set_title(f"{lab}: interaction coefficient across specifications")
fig.tight_layout(); fig.savefig(f"{FIG}/eval_spec_sensitivity.png", dpi=120); plt.close(fig)

print("Figures saved. Evaluation suite complete.")
