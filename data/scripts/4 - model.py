

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# LightGBM opcional
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


CRITICAL_CLASSES = {"preemergencia", "emergencia"}
GOOD_CLASS = "buena"


@dataclass
class ModelResult:
    horizon: int
    horizon_role: str
    name: str
    strategy: str
    condition_mode: str
    threshold_mode: str
    min_precision: float
    threshold: float

    # holdout metrics
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float

    # CV (train) metrics (robustness; optional)
    cv_roc_auc_mean: float
    cv_pr_auc_mean: float

    tn: int
    fp: int
    fn: int
    tp: int
    n_train: int
    n_test: int
    pos_train: int
    pos_test: int


def normalize_cat(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower().replace({
        "pre-emergencia": "preemergencia",
        "pre emerg": "preemergencia",
    }))


def make_binary_target(cat_future: pd.Series) -> np.ndarray:
    s = normalize_cat(cat_future)
    return s.isin(CRITICAL_CLASSES).astype(int).values


def temporal_split(df: pd.DataFrame, test_size: float):
    n = len(df)
    n_test = int(np.ceil(n * test_size))
    train = df.iloc[: n - n_test].copy()
    test = df.iloc[n - n_test :].copy()
    return train, test


def build_preprocess(numeric_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols)],
        remainder="drop"
    )


def make_calibrated(estimator, cv=3, method="sigmoid"):
    sig = inspect.signature(CalibratedClassifierCV.__init__)
    if "estimator" in sig.parameters:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def proba_from_pipeline(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise RuntimeError("Modelo sin predict_proba/decision_function.")


def pick_threshold(y_true, y_score, mode="max_recall_min_precision", min_precision=0.30) -> float:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best_t, best_metric = 0.5, -np.inf

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if mode == "max_f1":
            metric = f1
        elif mode == "max_recall_min_precision":
            metric = (r if p >= min_precision else -np.inf)
        else:
            raise ValueError("threshold_mode inválido")

        if metric > best_metric:
            best_metric = metric
            best_t = float(t)

    # fallback: si no alcanza precisión mínima, usa max_f1
    if mode == "max_recall_min_precision":
        if best_metric == -np.inf:
            return pick_threshold(y_true, y_score, mode="max_f1", min_precision=min_precision)
    return best_t


def eval_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pr_auc, roc_auc, precision, recall, f1, int(tn), int(fp), int(fn), int(tp)


def horizon_role(h: int) -> str:
    return "operational" if h in (24, 48) else "exploratory"


def apply_condition_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    if mode == "non_good":
        if "CAT" not in df.columns:
            raise ValueError("condition_mode=non_good requiere columna CAT (actual).")
        cat_now = normalize_cat(df["CAT"])
        return df.loc[cat_now.ne(GOOD_CLASS)].copy()
    raise ValueError("condition_mode inválido")


def candidate_models(random_state: int, pos_weight: float):
    models = []

    models.append(("LogisticRegression", "class_weight=balanced",
                   LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")))

    ridge = RidgeClassifier(class_weight="balanced", random_state=random_state)
    models.append(("RidgeClassifier", "calibrated+balanced",
                   make_calibrated(ridge, cv=3, method="sigmoid")))

    base_svm = LinearSVC(class_weight="balanced", random_state=random_state)
    models.append(("LinearSVC", "calibrated+balanced",
                   make_calibrated(base_svm, cv=3, method="sigmoid")))

    # Tree ensembles
    models.append(("RandomForest", "class_weight=balanced_subsample",
                   RandomForestClassifier(
                       n_estimators=700, random_state=random_state,
                       class_weight="balanced_subsample", n_jobs=-1,
                       min_samples_leaf=2
                   )))

    models.append(("ExtraTrees", "class_weight=balanced",
                   ExtraTreesClassifier(
                       n_estimators=900, random_state=random_state,
                       class_weight="balanced", n_jobs=-1,
                       min_samples_leaf=2
                   )))

    models.append(("HistGradientBoosting", "default",
                   HistGradientBoostingClassifier(
                       random_state=random_state, max_depth=6,
                       learning_rate=0.05, max_iter=600
                   )))

    # Classical boosting baselines (fast, robust)
    models.append(("GradientBoosting", "default",
                   GradientBoostingClassifier(random_state=random_state)))

    models.append(("AdaBoost", "default",
                   AdaBoostClassifier(random_state=random_state)))

    # Naive Bayes (baseline)
    models.append(("GaussianNB", "default", GaussianNB()))

    # XGBoost
    models.append(("XGBoost", f"scale_pos_weight={pos_weight:.2f}",
                   XGBClassifier(
                       n_estimators=900, learning_rate=0.03, max_depth=5,
                       subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, min_child_weight=1.0, gamma=0.0,
                       objective="binary:logistic", eval_metric="logloss",
                       scale_pos_weight=float(pos_weight), n_jobs=-1,
                       random_state=random_state
                   )))

    # LightGBM (opcional)
    if HAS_LGBM:
        models.append(("LightGBM", f"scale_pos_weight={pos_weight:.2f}",
                       LGBMClassifier(
                           n_estimators=1400, learning_rate=0.02,
                           num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, min_child_samples=20,
                           scale_pos_weight=float(pos_weight),
                           random_state=random_state, n_jobs=-1
                       )))

    # CatBoost
    w0 = 1.0
    w1 = float(pos_weight) if pos_weight > 1 else 1.0
    models.append(("CatBoost", f"class_weights=[{w0:.1f},{w1:.1f}]",
                   CatBoostClassifier(
                       iterations=1600, learning_rate=0.03, depth=6,
                       l2_leaf_reg=3.0, loss_function="Logloss",
                       eval_metric="AUC", random_seed=random_state,
                       verbose=False, class_weights=[w0, w1]
                   )))

    return models


def cv_auc_metrics(X_train, y_train, preprocess, model, tscv):
    """Promedios CV (threshold-free): ROC-AUC y PR-AUC en validaciones."""
    roc_list, pr_list = [], []
    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_tr, y_tr)
        s = proba_from_pipeline(pipe, X_va)
        if len(np.unique(y_va)) < 2:
            continue
        roc_list.append(roc_auc_score(y_va, s))
        pr_list.append(average_precision_score(y_va, s))
    return (float(np.mean(roc_list)) if roc_list else np.nan,
            float(np.mean(pr_list)) if pr_list else np.nan)


def train_one_horizon(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    target_col: str,
    test_size: float,
    splits: int,
    threshold_mode: str,
    min_precision: float,
    random_state: int,
    condition_mode: str
):
    dft = apply_condition_filter(df.copy(), condition_mode)
    dft = dft.sort_values("date").reset_index(drop=True)

    y = make_binary_target(dft[target_col])

    train_df, test_df = temporal_split(dft, test_size=test_size)
    X_train, y_train = train_df[feature_cols], y[:len(train_df)]
    X_test, y_test = test_df[feature_cols], y[len(train_df):]

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    preprocess = build_preprocess(feature_cols)
    tscv = TimeSeriesSplit(n_splits=splits)

    best = None
    best_score = -np.inf
    best_pipe = None

    all_res = []

    for name, strategy, model in candidate_models(random_state=random_state, pos_weight=pos_weight):
        # 1) CV threshold-free metrics (robustness)
        cv_roc, cv_pr = cv_auc_metrics(X_train, y_train, preprocess, model, tscv)

        # 2) OOF scores para seleccionar umbral
        oof_scores = np.zeros_like(y_train, dtype=float)
        oof_mask = np.zeros_like(y_train, dtype=bool)

        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr = y_train[tr_idx]

            pipe_fold = Pipeline([("preprocess", preprocess), ("model", model)])
            pipe_fold.fit(X_tr, y_tr)

            s_va = proba_from_pipeline(pipe_fold, X_va)
            oof_scores[va_idx] = s_va
            oof_mask[va_idx] = True

        y_oof = y_train[oof_mask]
        s_oof = oof_scores[oof_mask]
        if len(np.unique(y_oof)) < 2:
            continue

        thr = pick_threshold(y_oof, s_oof, mode=threshold_mode, min_precision=min_precision)

        # 3) Fit final + evaluación holdout
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)

        s_test = proba_from_pipeline(pipe, X_test)
        pr_auc, roc_auc, precision, recall, f1, tn, fp, fn, tp = eval_metrics(y_test, s_test, threshold=thr)

        res = ModelResult(
            horizon=horizon,
            horizon_role=horizon_role(horizon),
            name=name,
            strategy=strategy,
            condition_mode=condition_mode,
            threshold_mode=threshold_mode,
            min_precision=float(min_precision),
            threshold=float(thr),

            pr_auc=float(pr_auc) if pr_auc == pr_auc else np.nan,
            roc_auc=float(roc_auc) if roc_auc == roc_auc else np.nan,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),

            cv_roc_auc_mean=cv_roc,
            cv_pr_auc_mean=cv_pr,

            tn=tn, fp=fp, fn=fn, tp=tp,
            n_train=len(train_df),
            n_test=len(test_df),
            pos_train=int(np.sum(y_train == 1)),
            pos_test=int(np.sum(y_test == 1)),
        )
        all_res.append(res)

        # Score EWS: prioriza ROC/PR-AUC + recall
        score = (res.roc_auc if res.roc_auc == res.roc_auc else -1) \
                + 0.35 * (res.pr_auc if res.pr_auc == res.pr_auc else 0) \
                + 0.20 * res.recall \
                + 0.05 * res.f1
        if score > best_score:
            best_score = score
            best = res
            best_pipe = pipe

    if best is None:
        raise RuntimeError(f"Not found {horizon}h (cond={condition_mode}).")

    # preds del mejor
    X_test = test_df[feature_cols]
    best_scores = proba_from_pipeline(best_pipe, X_test)
    best_pred = (best_scores >= best.threshold).astype(int)

    preds = pd.DataFrame({
        "date": test_df["date"].values,
        "y_true": y_test,
        "y_score": best_scores,
        "y_pred": best_pred
    })

    return best, all_res, best_pipe, preds


def make_table1(df_metrics: pd.DataFrame, prefer_horizon=48, topk=1):

    out = []
    for h in sorted(df_metrics["horizon"].unique()):
        dh = df_metrics[df_metrics["horizon"] == h].copy()
        dh = dh.sort_values(["roc_auc", "recall", "pr_auc"], ascending=False)
        out.append(dh.head(topk))
    table = pd.concat(out, ignore_index=True)

    # columnas para el paper
    cols = [
        "name", "horizon", "roc_auc", "recall", "precision", "f1", "pr_auc",
        "threshold", "pos_test", "n_test", "condition_mode"
    ]
    table = table[cols].copy()
    table = table.rename(columns={
        "name": "Model",
        "horizon": "Horizon_h",
        "roc_auc": "ROC_AUC",
        "recall": "Recall_critical",
        "precision": "Precision",
        "f1": "F1",
        "pr_auc": "PR_AUC",
        "threshold": "Threshold",
        "pos_test": "Positives_test",
        "n_test": "N_test",
        "condition_mode": "Condition_mode"
    })
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset_modelo_limpio.csv",
                        help="CSV con: date, PM25,temp,hum,wind, CAT, CAT_24h, CAT_48h, CAT_72h")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--threshold_mode", type=str, default="max_recall_min_precision",
                        choices=["max_f1", "max_recall_min_precision"])
    parser.add_argument("--min_precision", type=float, default=0.30)
    parser.add_argument("--random_state", type=int, default=12)
    parser.add_argument("--skip_72h", action="store_true")
    parser.add_argument("--condition_mode", type=str, default="all",
                        choices=["all", "non_good"])
    parser.add_argument("--topk", type=int, default=1,
                        help="Para Table 1: top-k modelos por horizonte.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"No encuentro: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        raise ValueError("El dataset debe tener columna 'date'.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feature_cols = ["PM25", "wind", "temp", "hum"]
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan features: {miss}")

    target_cols = {24: "CAT_24h", 48: "CAT_48h", 72: "CAT_72h"}
    if args.skip_72h:
        target_cols = {24: "CAT_24h", 48: "CAT_48h"}
    for h, col in target_cols.items():
        if col not in df.columns:
            raise ValueError(f"Falta columna objetivo: {col}")

    models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.condition_mode == "all" else f"_cond_{args.condition_mode}"

    all_metrics_rows = []
    best_rows = []

    for h, tcol in target_cols.items():
        best, all_results, best_pipe, preds = train_one_horizon(
            df=df,
            horizon=h,
            feature_cols=feature_cols,
            target_col=tcol,
            test_size=args.test_size,
            splits=args.splits,
            threshold_mode=args.threshold_mode,
            min_precision=args.min_precision,
            random_state=args.random_state,
            condition_mode=args.condition_mode
        )

    
        model_out = models_dir / f"best_{h}h{suffix}.joblib"
        dump({
            "model": best_pipe,
            "threshold": best.threshold,
            "threshold_mode": best.threshold_mode,
            "min_precision": best.min_precision,
            "feature_cols": feature_cols,
            "horizon_hours": h,
            "condition_mode": args.condition_mode,
            "target_col": tcol,
            "split": {"type": "temporal_holdout", "test_size": args.test_size},
            "cv": {"type": "TimeSeriesSplit", "splits": args.splits},
        }, model_out)

        preds_out = reports_dir / f"test_predictions_{h}h{suffix}.csv"
        preds.to_csv(preds_out, index=False)

        best_rows.append(asdict(best))
        all_metrics_rows.extend([asdict(r) for r in all_results])

    df_all = pd.DataFrame(all_metrics_rows)
    df_best = pd.DataFrame(best_rows)

    metrics_all_out = reports_dir / f"metrics_all_models{suffix}.csv"
    metrics_best_out = reports_dir / f"metrics_best_by_horizon{suffix}.csv"
    df_all.to_csv(metrics_all_out, index=False)
    df_best.to_csv(metrics_best_out, index=False)

    table1 = make_table1(df_all, topk=args.topk)
    table1_out = reports_dir / f"table1_models_by_horizon_top{args.topk}{suffix}.csv"
    table1.to_csv(table1_out, index=False)


    top48 = df_all[df_all["horizon"] == 48].sort_values(["roc_auc", "recall"], ascending=False)
    top48_out = reports_dir / f"table1_top_models_48h{suffix}.csv"
    top48.head(15).to_csv(top48_out, index=False)


    print("All models:", metrics_all_out)
    print("Best per horizon:", metrics_best_out)
    print("Table 1:", table1_out)
    print("Top 48h:", top48_out)


if __name__ == "__main__":
    main()
