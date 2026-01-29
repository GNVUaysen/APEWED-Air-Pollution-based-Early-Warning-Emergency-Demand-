#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Association between air pollution episodes and daily emergency room demand
Coyhaique, Chile (2015–2025)

Outcomes:
- TOTAL_URG  : Total emergency room visits
- RESP_TOTAL : Respiratory emergency visits (J00–J98)

Methods:
- Descriptive statistics by air-quality category (CAT)
- Non-parametric tests (Kruskal–Wallis, Spearman)
- Negative binomial regression (counts)
- Rate model with offset log(TOTAL_URG)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# Paths
# =========================
PATH_URGENCIAS = "Atenciones_Urgencia_Consolidado.xlsx"
PATH_AIR       = "dataset_modelo_limpio.csv"

# =========================
# Categories of interest
# =========================
CATS = {
    "TOTAL_URG":  "SECCIÓN 1. TOTAL ATENCIONES DE URGENCIA",
    "RESP_TOTAL": "TOTAL CAUSA SISTEMA  RESPIRATORIO (J00-J98)",
}

CAT_ORDER = ["buena", "regular", "alerta", "preemergencia", "emergencia"]
CAT_MAP = {c: i for i, c in enumerate(CAT_ORDER)}

# ==================================================
# Read emergency data (all sheets, daily series)
# ==================================================
def read_urgencias(excel_path):
    xls = pd.ExcelFile(excel_path)
    frames = []
    inv = {v: k for k, v in CATS.items()}

    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df = df.rename(columns={df.columns[0]: "CATEGORIA"})
        df = df[df["CATEGORIA"].isin(CATS.values())]

        if df.empty:
            continue

        date_cols = [c for c in df.columns if c != "CATEGORIA"]
        long = df.melt("CATEGORIA", date_cols,
                       var_name="FECHA_RAW", value_name="VALOR")

        long["FECHA"] = pd.to_datetime(long["FECHA_RAW"], errors="coerce")
        long["VALOR"] = pd.to_numeric(long["VALOR"].replace("-", np.nan),
                                      errors="coerce")
        long["VAR"] = long["CATEGORIA"].map(inv)

        frames.append(long[["FECHA", "VAR", "VALOR"]])

    data = pd.concat(frames).dropna(subset=["FECHA"])
    daily = (data
             .pivot_table(index="FECHA", columns="VAR",
                          values="VALOR", aggfunc="sum")
             .reset_index()
             .sort_values("FECHA"))

    return daily

# ==================================================
# Read air-quality data
# ==================================================
def read_air(csv_path):
    air = pd.read_csv(csv_path)
    air["FECHA"] = pd.to_datetime(air["date"], errors="coerce")
    air["CAT"] = air["CAT"].str.strip().str.lower()
    air["CAT_ORD"] = air["CAT"].map(CAT_MAP)
    return air[["FECHA", "CAT", "CAT_ORD"]].dropna()

# ==================================================
# IRR table
# ==================================================
def irr_table(model):
    ci = model.conf_int()
    return pd.DataFrame({
        "IRR": np.exp(model.params),
        "CI95_L": np.exp(ci[0]),
        "CI95_U": np.exp(ci[1]),
        "p": model.pvalues
    })

# ==================================================
# Main analysis
# ==================================================
def main():
    urg = read_urgencias(PATH_URGENCIAS)
    air = read_air(PATH_AIR)

    df = pd.merge(urg, air, on="FECHA", how="inner")
    df = df.sort_values("FECHA")

    # Controls
    df["dow"] = df["FECHA"].dt.dayofweek
    df["month"] = df["FECHA"].dt.month
    df["t"] = np.arange(len(df))
    df["CAT"] = pd.Categorical(df["CAT"], CAT_ORDER, ordered=True)

    # ===============================
    # A) Descriptive statistics
    # ===============================
    desc = df.groupby("CAT")[["TOTAL_URG", "RESP_TOTAL"]] \
             .agg(["count", "mean", "median"])
    desc.to_csv("descriptivos_por_CAT.csv")

    # ===============================
    # B) Non-parametric tests
    # ===============================
    for y in ["TOTAL_URG", "RESP_TOTAL"]:
        groups = [df.loc[df["CAT"] == c, y].dropna()
                  for c in CAT_ORDER if (df["CAT"] == c).sum() > 20]
        kw = stats.kruskal(*groups)
        print(f"Kruskal–Wallis {y}: H={kw.statistic:.2f}, p={kw.pvalue:.3g}")

        rho, p = stats.spearmanr(df["CAT_ORD"], df[y], nan_policy="omit")
        print(f"Spearman {y}: rho={rho:.2f}, p={p:.3g}")

    # ===============================
    # C) Negative binomial models
    # ===============================
    formula = "C(CAT) + C(dow) + C(month) + t"

    for y in ["TOTAL_URG", "RESP_TOTAL"]:
        m = smf.glm(f"{y} ~ {formula}", data=df,
                    family=sm.families.NegativeBinomial()).fit()
        irr = irr_table(m)
        irr.loc[irr.index.str.contains("C\\(CAT\\)")].to_csv(f"IRR_NB_{y}.csv")

    # ===============================
    # D) Rate model (offset)
    # ===============================
    df_rate = df[df["TOTAL_URG"] > 0].copy()
    df_rate["log_total"] = np.log(df_rate["TOTAL_URG"])

    m_rate = smf.glm(
        f"RESP_TOTAL ~ {formula}",
        data=df_rate,
        family=sm.families.NegativeBinomial(),
        offset=df_rate["log_total"]
    ).fit()

    irr_rate = irr_table(m_rate)
    irr_rate.loc[irr_rate.index.str.contains("C\\(CAT\\)")].to_csv(
        "IRR_RATE_RESP.csv"
    )

    df.to_csv("dataset_final_urgencias_aire.csv", index=False)

if __name__ == "__main__":
    main()

