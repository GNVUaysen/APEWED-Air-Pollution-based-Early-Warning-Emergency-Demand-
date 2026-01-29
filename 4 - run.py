#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

FEATURES = ["PM25", "wind", "temp", "hum"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path al .joblib guardado por model.py (p.ej., models/best_48h.joblib)")
    ap.add_argument("--input_csv", required=True, help="CSV con una fila y columnas PM25, wind, temp, hum")
    ap.add_argument("--out_json", default="", help="Opcional: ruta para guardar salida JSON")
    args = ap.parse_args()

    bundle = load(args.model)
    pipe = bundle["model"]
    thr  = float(bundle["threshold"])
    meta = {k: bundle.get(k) for k in ["horizon_hours", "target_col", "condition_mode", "feature_cols"]}

    x = pd.read_csv(args.input_csv)
    missing = [c for c in FEATURES if c not in x.columns]
    if missing:
        raise ValueError(f"Faltan columnas en input: {missing}")

    X = x[FEATURES].copy()

    # Probabilidad de crítico (preemergencia/emergencia)
    if hasattr(pipe, "predict_proba"):
        p = float(pipe.predict_proba(X)[:, 1][0])
    else:
        # fallback si el clasificador no expone proba (poco probable en tu pipeline)
        s = float(pipe.decision_function(X)[0])
        p = float(1 / (1 + np.exp(-s)))

    yhat = int(p >= thr)

    out = {
        "prob_critical_48h": p,
        "pred_critical_48h": yhat,
        "threshold": thr,
        "meta": meta,
        "interpretation": (
            "High risk of pre-emergency/emergency in 48h" if yhat == 1 else "Low risk of critical pollution in 48h"
        )
    }

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)

    if args.out_json:
        Path(args.out_json).write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    main()
