# -*- coding: utf-8 -*-
import os, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# =========================
# 공용 상수
# =========================
COL_DATE   = "일자"
COL_REGION = "지역명"

CANDIDATE_COLS = [
    "평균기온","최고기온","최저기온",
    "습도","강우량","일사량",
    "결로시간","평균풍속","최대풍속"
]

KEY_COLS = {"지역명","품종","일자","계측시간","year","month"}

CORR_TOP_K    = 25
CORR_MIN_ABS  = 0.20
VIF_THRESHOLD = 10.0
LASSO_ALPHAS  = np.logspace(-3, 1, 60)
LASSO_FOLDS   = 5
RANDOM_STATE  = 42

CULTIVAR_MONTH_WINDOWS = {
    "홍로": list(range(4, 9)),   # 4~8월
    "후지": list(range(4, 11)),  # 4~10월
}

def _norm_cultivar_name(name):
    return str(name).strip() if pd.notna(name) else None

# =========================
# 유틸
# =========================
def to_year(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.year

_M_SUFFIX_RE = re.compile(r"_m(\d{2})$")

def _extract_month_from_col(col: str):
    m = _M_SUFFIX_RE.search(col)
    return int(m.group(1)) if m else None

def filter_columns_by_month(df: pd.DataFrame, allowed_months: list[int]) -> pd.DataFrame:
    if not allowed_months:
        return df
    keep_cols = []
    for c in df.columns:
        if c in {COL_REGION, "year"}:
            keep_cols.append(c); continue
        mm = _extract_month_from_col(c)
        if mm is None or mm in allowed_months:
            keep_cols.append(c)
    return df[keep_cols]

# =========================
# 특징 생성
# =========================
def build_monthly_features(env_daily: pd.DataFrame, env_cols):
    df = env_daily[[COL_REGION, "year", "month"] + env_cols].copy()
    mean_like = [c for c in env_cols if re.search("기온|습도|풍속", c)]
    sum_like  = [c for c in env_cols if re.search("일사|강우|강수|복사|rad|precip", c, re.I)]
    remainder = [c for c in env_cols if c not in mean_like + sum_like]
    mean_like += remainder
    agg_map = {c: "mean" for c in mean_like}
    agg_map.update({c: "sum" for c in sum_like})
    g = df.groupby([COL_REGION, "year", "month"], as_index=False).agg(agg_map)

    wide = None
    for m in range(1, 13):
        sub = g[g["month"] == m].drop(columns=["month"])
        rename = {c: f"{c}_{agg_map.get(c,'mean')}_m{m:02d}" for c in env_cols}
        sub = sub.rename(columns=rename)
        wide = sub if wide is None else pd.merge(wide, sub, on=[COL_REGION, "year"], how="outer")
    return wide

# =========================
# 변수 선택 & 모델링
# =========================
def select_by_correlation(df_merged, target, top_k=CORR_TOP_K, min_abs=CORR_MIN_ABS):
    num_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in {"year"} and c != target]
    corr = df_merged[feats + [target]].corr(numeric_only=True)[target].drop(target)
    corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    kept = corr[abs(corr) >= min_abs].index.tolist()
    if len(kept) > top_k:
        kept = kept[:top_k]
    return kept, corr

def lasso_shrink(X, y, alphas=LASSO_ALPHAS, folds=LASSO_FOLDS, random_state=RANDOM_STATE):
    Xs = (X - X.mean()) / X.std(ddof=0)
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    lcv = LassoCV(alphas=alphas, cv=cv, random_state=random_state, max_iter=20000).fit(Xs, y)
    coef = pd.Series(lcv.coef_, index=X.columns)
    selected = coef[coef.abs() > 1e-8].index.tolist()
    return selected, coef, float(lcv.alpha_)

def vif_prune(X, thresh=VIF_THRESHOLD):
    feats = list(X.columns)
    changed = True
    while changed and len(feats) > 1:
        changed = False
        Xc = sm.add_constant(X[feats])
        vifs = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
        vif_s = pd.Series(vifs, index=["const"] + feats).drop("const")
        if vif_s.max() > thresh:
            drop = vif_s.idxmax()
            feats.remove(drop)
            changed = True
    return feats

def finalize_ols(X, y):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit()

def equation_string(model, features, target):
    p = model.params
    parts = [f"{p['const']:.6g}"] + [f"{p.get(f,0):+.6g}·{f}" for f in features]
    return f"{target} = " + " ".join(parts)

def run_single_target(env_daily, fruit_sub, env_cols, target_col, cultivar_name=None):
    monthly = build_monthly_features(env_daily, env_cols)
    allowed_months = None
    if cultivar_name is not None:
        cv = _norm_cultivar_name(cultivar_name)
        if cv in CULTIVAR_MONTH_WINDOWS:
            allowed_months = CULTIVAR_MONTH_WINDOWS[cv]
    if allowed_months:
        monthly = filter_columns_by_month(monthly, allowed_months)

    merged = pd.merge(fruit_sub, monthly, on=[COL_REGION, "year"], how="inner").dropna(subset=[target_col])

    if not pd.api.types.is_numeric_dtype(merged[target_col]):
        merged[target_col] = pd.to_numeric(merged[target_col], errors="coerce")
    merged = merged.dropna(subset=[target_col])

    kept, corr_s = select_by_correlation(merged, target_col)
    env_keywords = ("기온", "습도", "강우", "일사", "결로", "풍속")
    kept = [c for c in kept if any(k in c for k in env_keywords)]
    if len(kept) == 0:
        raise RuntimeError(f"{target_col}: 선택된 환경 변수가 없습니다")

    X = merged[kept].astype(float).fillna(merged[kept].mean())
    y = merged[target_col].astype(float)

    sel, lasso_coef, alpha = lasso_shrink(X, y)
    if len(sel) == 0:
        sel = kept[:min(5, len(kept))]
    sel_vif = vif_prune(X[sel], VIF_THRESHOLD)
    features = sel_vif if len(sel_vif) > 0 else sel

    model = finalize_ols(X[features], y)
    eq = equation_string(model, features, target_col)

    yhat = model.predict(sm.add_constant(X[features]))
    r2 = float(r2_score(y, yhat))
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))

    summary = {
        "n_samples": int(len(merged)),
        "r2_in_sample": round(r2, 3),
        "rmse_in_sample": round(rmse, 3),
        "lasso_alpha": float(alpha),
        "features": features,
        "equation": eq,
        "target": target_col,
        "cultivar": cultivar_name
    }
    return model, summary, corr_s, merged

# =========================
# 모델 저장/불러오기 (레지스트리)
# =========================
def save_model_registry(model_dir: Path, model, meta: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver_dir = model_dir / f"v_{ts}"
    ver_dir.mkdir()
    joblib.dump(model, ver_dir / "model.joblib")
    with open(ver_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return ver_dir

def load_latest_model(model_dir: Path):
    if not model_dir.exists():
        return None, None
    versions = sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("v_")])
    if not versions:
        return None, None
    latest = versions[-1]
    model = joblib.load(latest / "model.joblib")
    with open(latest / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta
