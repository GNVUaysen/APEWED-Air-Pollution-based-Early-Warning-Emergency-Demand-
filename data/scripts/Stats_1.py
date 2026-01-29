import pandas as pd
import numpy as np
from pathlib import Path

FILE = Path("Total_ER.xlsx")

DATE_COL = "FECHA"
RESP_COL = "RESP_TOTAL"
TOT_COL = "TOTAL_URG"
CAT_COL = "CAT"

WINTER_DEF = "may_aug"
WINTER_MONTHS = set([5, 6, 7, 8]) if WINTER_DEF == "may_aug" else set([4, 5, 6, 7, 8, 9])

CRITICAL_CATS = ["preemergencia", "emergencia"]
MIN_CLUSTER_LEN = 2
MAX_LAG = 6
BASELINE_LAGS = [-3, -2, -1]

BOOT = 2000
RNG_SEED = 42

def to_numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace(["-", ""], np.nan), errors="coerce")

def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, (np.nan, np.nan)
    diffs = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    diffs = np.array(diffs)
    return (np.mean(a) - np.mean(b)), (np.quantile(diffs, 0.025), np.quantile(diffs, 0.975))

def permutation_test_mean_diff(a: np.ndarray, b: np.ndarray, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    obs = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])
        if abs(diff) >= abs(obs):
            count += 1
    return (count + 1) / (n + 1)

def summarize_series(x: pd.Series):
    x = x.dropna()
    return {
        "n": int(x.shape[0]),
        "mean": float(x.mean()) if len(x) else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "p25": float(x.quantile(0.25)) if len(x) else np.nan,
        "p75": float(x.quantile(0.75)) if len(x) else np.nan,
        "p90": float(x.quantile(0.90)) if len(x) else np.nan,
    }

def load_excel_consolidado(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    dfs = []
    required = {DATE_COL, RESP_COL, TOT_COL, CAT_COL}
    for sheet in xls.sheet_names:
        d = pd.read_excel(xls, sheet_name=sheet)
        d = d.dropna(how="all")
        d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
        if not required.issubset(set(d.columns)):
            continue
        dfs.append(d)
    if not dfs:
        raise ValueError("Columns not found")
    return pd.concat(dfs, ignore_index=True)

df = load_excel_consolidado(FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[RESP_COL] = to_numeric_clean(df[RESP_COL])
df[TOT_COL] = to_numeric_clean(df[TOT_COL])

df = df.dropna(subset=[DATE_COL, RESP_COL, TOT_COL, CAT_COL])
df = df[(df[TOT_COL] > 0) & (df[RESP_COL] >= 0)]
df = df.sort_values(DATE_COL).reset_index(drop=True)

df["year"] = df[DATE_COL].dt.year
df["month"] = df[DATE_COL].dt.month
df["dow"] = df[DATE_COL].dt.dayofweek
df["is_winter"] = df["month"].isin(WINTER_MONTHS)

df["resp_share"] = df[RESP_COL] / df[TOT_COL]

desc_rows = []
for name, mask in [
    ("all_days", pd.Series(True, index=df.index)),
    ("winter", df["is_winter"]),
    ("non_winter", ~df["is_winter"]),
]:
    d = df[mask]
    row = {
        "group": name,
        **{f"resp_{k}": v for k, v in summarize_series(d[RESP_COL]).items()},
        **{f"tot_{k}": v for k, v in summarize_series(d[TOT_COL]).items()},
        **{f"share_{k}": v for k, v in summarize_series(d["resp_share"]).items()},
    }
    desc_rows.append(row)

desc_overall = pd.DataFrame(desc_rows)

w = df[df["is_winter"]][RESP_COL].to_numpy()
nw = df[~df["is_winter"]][RESP_COL].to_numpy()

obs_diff_resp, ci_resp = bootstrap_mean_diff(w, nw, n=BOOT, seed=RNG_SEED)
p_perm_resp = permutation_test_mean_diff(w, nw, n=5000, seed=RNG_SEED)

wS = df[df["is_winter"]]["resp_share"].to_numpy()
nwS = df[~df["is_winter"]]["resp_share"].to_numpy()

obs_diff_share, ci_share = bootstrap_mean_diff(wS, nwS, n=BOOT, seed=RNG_SEED)
p_perm_share = permutation_test_mean_diff(wS, nwS, n=5000, seed=RNG_SEED)

winter_tests = pd.DataFrame([{
    "winter_def": WINTER_DEF,
    "diff_mean_resp_w_minus_nw": obs_diff_resp,
    "boot95_low_resp": ci_resp[0],
    "boot95_high_resp": ci_resp[1],
    "perm_pvalue_resp": p_perm_resp,
    "diff_mean_share_w_minus_nw": obs_diff_share,
    "boot95_low_share": ci_share[0],
    "boot95_high_share": ci_share[1],
    "perm_pvalue_share": p_perm_share,
}])

df["is_critical"] = df[CAT_COL].isin(CRITICAL_CATS)
df["run_id"] = (df["is_critical"] != df["is_critical"].shift()).cumsum()

clusters = (
    df[df["is_critical"]]
    .groupby("run_id")
    .agg(
        start_date=(DATE_COL, "min"),
        end_date=(DATE_COL, "max"),
        length=(DATE_COL, "count"),
        max_cat=(CAT_COL, lambda x: "emergencia" if "emergencia" in x.values else "preemergencia"),
    )
    .reset_index()
    .rename(columns={"run_id": "cluster_id"})
)

clusters = clusters[clusters["length"] >= MIN_CLUSTER_LEN].reset_index(drop=True)

events = []
df_idx = df.set_index(DATE_COL)

for _, c in clusters.iterrows():
    end_day = c["end_date"]
    baseline_vals = []
    for lag in BASELINE_LAGS:
        day = end_day + pd.Timedelta(days=lag)
        if day in df_idx.index:
            baseline_vals.append(df_idx.loc[day, RESP_COL])
    baseline_mean = np.mean(baseline_vals) if len(baseline_vals) else np.nan
    for lag in range(0, MAX_LAG + 1):
        day = end_day + pd.Timedelta(days=lag)
        if day in df_idx.index:
            resp_val = float(df_idx.loc[day, RESP_COL])
            events.append({
                "cluster_id": int(c["cluster_id"]),
                "cluster_length": int(c["length"]),
                "cluster_max_cat": c["max_cat"],
                "end_date": end_day,
                "lag": int(lag),
                "RESP_TOTAL": resp_val,
                "baseline_mean": baseline_mean,
                "delta_vs_baseline": resp_val - baseline_mean if not np.isnan(baseline_mean) else np.nan
            })

event_df = pd.DataFrame(events)

summary_rows = []
rng = np.random.default_rng(RNG_SEED)

for lag in range(0, MAX_LAG + 1):
    dlag = event_df[event_df["lag"] == lag]["delta_vs_baseline"].dropna().to_numpy()
    if len(dlag) == 0:
        summary_rows.append({"lag": lag, "n_obs": 0})
        continue
    boots = []
    for _ in range(BOOT):
        sample = rng.choice(dlag, size=len(dlag), replace=True)
        boots.append(np.mean(sample))
    boots = np.array(boots)
    summary_rows.append({
        "lag": lag,
        "n_obs": int(len(dlag)),
        "mean_delta": float(np.mean(dlag)),
        "median_delta": float(np.median(dlag)),
        "boot95_low": float(np.quantile(boots, 0.025)),
        "boot95_high": float(np.quantile(boots, 0.975)),
    })

event_summary = pd.DataFrame(summary_rows)

cum_rows = []
for cid, g in event_df.dropna(subset=["delta_vs_baseline"]).groupby("cluster_id"):
    g_post = g[(g["lag"] >= 1) & (g["lag"] <= MAX_LAG)]
    cum_delta = g_post["delta_vs_baseline"].sum()
    peak_lag = int(g_post.loc[g_post["delta_vs_baseline"].idxmax(), "lag"]) if not g_post.empty else np.nan
    peak_delta = float(g_post["delta_vs_baseline"].max()) if not g_post.empty else np.nan
    meta = g.iloc[0]
    cum_rows.append({
        "cluster_id": int(cid),
        "end_date": meta["end_date"],
        "cluster_length": int(meta["cluster_length"]),
        "cluster_max_cat": meta["cluster_max_cat"],
        "cum_delta_lag1_to_lag6": float(cum_delta),
        "peak_lag_1to6": peak_lag,
        "peak_delta_1to6": peak_delta
    })

cum_impact = pd.DataFrame(cum_rows)

def length_bin(x):
    if x == 2: return "2"
    if x == 3: return "3"
    if x == 4: return "4"
    if x >= 5: return "5+"
    return "other"

cum_impact["length_bin"] = cum_impact["cluster_length"].apply(length_bin)

cum_by_group = (
    cum_impact
    .groupby(["cluster_max_cat", "length_bin"])
    .agg(
        n_clusters=("cluster_id", "count"),
        mean_cum_delta=("cum_delta_lag1_to_lag6", "mean"),
        median_cum_delta=("cum_delta_lag1_to_lag6", "median"),
        mean_peak_delta=("peak_delta_1to6", "mean"),
        mean_peak_lag=("peak_lag_1to6", "mean"),
    )
    .reset_index()
)

exposed = pd.Series(False, index=df.index)
date_to_i = {d: i for i, d in enumerate(df[DATE_COL])}

for _, c in clusters.iterrows():
    start = c["start_date"]
    end = c["end_date"]
    in_cluster = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
    exposed |= in_cluster
    for lag in range(0, MAX_LAG + 1):
        day = end + pd.Timedelta(days=lag)
        if day in date_to_i:
            exposed.iloc[date_to_i[day]] = True

df["is_exposed_cluster_or_post"] = exposed

df_ne = df[~df["is_exposed_cluster_or_post"]]
w_ne = df_ne[df_ne["is_winter"]][RESP_COL].to_numpy()
nw_ne = df_ne[~df_ne["is_winter"]][RESP_COL].to_numpy()

obs_diff_resp_ne, ci_resp_ne = bootstrap_mean_diff(w_ne, nw_ne, n=BOOT, seed=RNG_SEED)
p_perm_resp_ne = permutation_test_mean_diff(w_ne, nw_ne, n=5000, seed=RNG_SEED)

winter_tests_excluding_exposed = pd.DataFrame([{
    "winter_def": WINTER_DEF,
    "note": "winter vs non-winter excluding cluster days + post window",
    "diff_mean_resp_w_minus_nw": obs_diff_resp_ne,
    "boot95_low_resp": ci_resp_ne[0],
    "boot95_high_resp": ci_resp_ne[1],
    "perm_pvalue_resp": p_perm_resp_ne,
    "n_days_used": int(df_ne.shape[0])
}])

desc_overall.to_csv("A_desc_overall_groups.csv", index=False)
winter_tests.to_csv("B_winter_tests_bootstrap_perm.csv", index=False)
clusters.to_csv("C_clusters_high_pollution.csv", index=False)
event_df.to_csv("D_event_study_resp_clusters.csv", index=False)
event_summary.to_csv("E_event_study_summary_by_lag.csv", index=False)
cum_impact.to_csv("F_cumulative_impact_by_cluster.csv", index=False)
cum_by_group.to_csv("G_cumulative_impact_by_length_severity.csv", index=False)
winter_tests_excluding_exposed.to_csv("H_winter_tests_excluding_exposed.csv", index=False)

print("\n===  groups (all/winter/non-winter) ===")
print(desc_overall.round(4).to_string(index=False))

print("\n=== Winter tests (RESP_TOTAL & share) ===")
print(winter_tests.round(6).to_string(index=False))

print("\n=== Clusters ===")
print(clusters.head().to_string(index=False))
print("\nN clusters:", clusters.shape[0], "| mean length:", clusters["length"].mean())

print("\n=== Event study summary by lag (delta vs baseline) ===")
print(event_summary.round(4).to_string(index=False))

print("\n=== Cumulative impact (by length & severity) ===")
print(cum_by_group.round(4).to_string(index=False))

print("\n=== Winter tests excluding exposed days (supports 'not seasonality per se') ===")
print(winter_tests_excluding_exposed.round(6).to_string(index=False))

