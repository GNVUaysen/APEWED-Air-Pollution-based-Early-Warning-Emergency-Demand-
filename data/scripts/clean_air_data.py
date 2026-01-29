from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(".")
FILES = {
    "PM25": "MP2.5.csv",
    "PM10": "MP10.csv",
    "temp": "temp.csv",
    "hum": "hum.csv",
    "pres": "pres.csv",
    "wind": "wind.csv",
    "wind_dir": "wind_dir.csv",
}

OUT_RAW = "dataset_model_clean_raw.csv"
OUT_SCALED = "dataset_model_clean.csv"
OUT_SCALER = "scaler.npy"

OUTLIER_Q_LOW = 0.01
OUTLIER_Q_HIGH = 0.99


def read_csv_robust(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path, sep=";", engine="python", dtype=str)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_date_YYMMDD(series: pd.Series) -> pd.Series:

    series = series.astype(str).str.strip()

    def convert(x: str):
        x = "".join(ch for ch in x if ch.isdigit())
        if len(x) != 6:
            return np.nan
        yy = int(x[:2])
        yyyy = 2000 + yy if yy < 70 else 1900 + yy
        mm = x[2:4]
        dd = x[4:6]
        return f"{yyyy}-{mm}-{dd}"

    return pd.to_datetime(series.map(convert), errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:

    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_variable(path: Path, var_name: str) -> pd.DataFrame:

    df = read_csv_robust(path)


    date_candidates = [c for c in df.columns if "FECHA" in c.upper()]
    if not date_candidates:
        raise ValueError(f"[{path.name}] Not found FECHA")
    date_col = date_candidates[0]


    hour_candidates = [c for c in df.columns if "HORA" in c.upper()]
    has_hour = len(hour_candidates) > 0
    hour_col = hour_candidates[0] if has_hour else None

    df["date"] = parse_date_YYMMDD(df[date_col])

    ignore = set([date_col] + ([hour_col] if hour_col else []))
    value_cols = [c for c in df.columns if c not in ignore and c != "date"]

    if not value_cols:
        raise ValueError(f"[{path.name}] not found")

    best_col = None
    best_count = -1
    best_series = None
    for c in value_cols:
        s = to_numeric(df[c])
        cnt = int(s.notna().sum())
        if cnt > best_count:
            best_count = cnt
            best_col = c
            best_series = s

    if best_col is None or best_count == 0:
        raise ValueError(f"[{path.name}] not found")

    out = pd.DataFrame({"date": df["date"], var_name: best_series}).dropna(subset=["date"])
    out = out.set_index("date").sort_index()

    # Info rango
    fmin = out.index.min()
    fmax = out.index.max()
    print(f"{path.name:18s} | {fmin.date()} → {fmax.date()} | {'HORARIO' if has_hour else 'DIARIO'}")

    # Agregación diaria
    out = out.resample("D").mean()

    return out


def remove_outliers(df: pd.DataFrame, cols) -> pd.DataFrame:

    out = df.copy()
    for c in cols:
        ql = out[c].quantile(OUTLIER_Q_LOW)
        qh = out[c].quantile(OUTLIER_Q_HIGH)
        out = out[(out[c] >= ql) & (out[c] <= qh)]
    return out


def air_quality_category(pm25: float) -> str:

    if pm25 > 169:
        return "emergencia"
    elif pm25 > 109:
        return "preemergencia"
    elif pm25 > 79:
        return "alerta"
    elif pm25 > 50:
        return "regular"
    else:
        return "buena"


# =========================
# MAIN
# =========================
def main():
    print("\n=== day range ===")
    frames = []
    for var, fname in FILES.items():
        fpath = (DATA_DIR / fname).resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Not found: {fpath}")
        frames.append(load_variable(fpath, var))

 
    data = pd.concat(frames, axis=1, join="inner").sort_index()


    data = data.interpolate(method="time", limit_direction="both")


    data = data.dropna()


    feature_cols = list(FILES.keys())
    data = remove_outliers(data, feature_cols)

    data["CAT"] = data["PM25"].apply(air_quality_category)
    data["CAT_24h"] = data["CAT"].shift(-1)
    data["CAT_48h"] = data["CAT"].shift(-2)
    data["CAT_72h"] = data["CAT"].shift(-3)

    data = data.dropna(subset=["CAT_24h", "CAT_48h", "CAT_72h"])


    raw = data.reset_index().rename(columns={"index": "date"})
    raw.to_csv(OUT_RAW, index=False)


    scaler = StandardScaler()
    X = scaler.fit_transform(data[feature_cols].astype(float).values)

    scaled = pd.DataFrame(X, columns=feature_cols, index=data.index)
    final = pd.concat([scaled, data[["CAT", "CAT_24h", "CAT_48h", "CAT_72h"]]], axis=1)
    final = final.reset_index().rename(columns={"index": "date"})
    final.to_csv(OUT_SCALED, index=False)


    np.save(OUT_SCALER, {"mean": scaler.mean_, "std": scaler.scale_, "features": feature_cols}, allow_pickle=True)

    print("\n=== FINAL DATASET ===")
    print("Range:", final["date"].min().date(), "→", final["date"].max().date())
    print("Rows:", final.shape[0], " | Columns:", final.shape[1])
    print("\nDistribution CAT (t0):")
    print(final["CAT"].value_counts())


    print(f"  - {OUT_RAW}")
    print(f"  - {OUT_SCALED}")
    print(f"  - {OUT_SCALER}")


if __name__ == "__main__":
    main()

