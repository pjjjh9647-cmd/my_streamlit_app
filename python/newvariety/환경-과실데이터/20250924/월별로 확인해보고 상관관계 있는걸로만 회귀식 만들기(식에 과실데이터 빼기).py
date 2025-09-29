# -*- coding: utf-8 -*-
import os, re
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

# =========================
# 사용자 입력
# =========================
ENV_DAILY_PATH    = r"C:\Users\User\Desktop\환경데이터\기상데이터_통합.xlsx"   # 일별 기상
FRUIT_TARGET_PATH = r"C:\Users\User\Desktop\과실데이터\과실데이터_통합.xlsx" # 과실 지표
OUTDIR            = Path(r"C:\Users\User\Desktop\분석결과")

COL_DATE   = "일자"     # 기상 날짜 컬럼명
COL_REGION = "지역명"   # 지역 키

# 환경 후보(존재하는 것만 자동 사용)
CANDIDATE_COLS = [
    "평균기온","최고기온","최저기온",
    "습도","강우량","일사량",
    "결로시간","평균풍속","최대풍속"
]

# 타깃 자동 추출 시 제외할 키
KEY_COLS = {"지역명","품종","일자","계측시간","year","month"}

# 선택/규제/진단 파라미터
CORR_TOP_K    = 25
CORR_MIN_ABS  = 0.20
VIF_THRESHOLD = 10.0
LASSO_ALPHAS  = np.logspace(-3, 1, 60)
LASSO_FOLDS   = 5
RANDOM_STATE  = 42

INCLUDE_MERGED = True       # 병합 원천 시트 포함
GROUP_BY_CULTIVAR = True    # 품종별로 따로 모델링(과실데이터에 '품종' 없으면 자동 통합)

# =========================
# 파일 저장: 안전 컨텍스트 매니저
# =========================
@contextmanager
def safe_excel_writer(dest: Path):
    """엑셀이 열려 있어 PermissionError가 나면 타임스탬프 붙여 다른 이름으로 저장."""
    try:
        writer = pd.ExcelWriter(dest, engine="xlsxwriter")
        yield writer
        writer.close()
        print("저장 완료:", dest)
    except PermissionError:
        alt = dest.with_name(f"{dest.stem}_{datetime.now():%Y%m%d_%H%M%S}{dest.suffix}")
        writer = pd.ExcelWriter(alt, engine="xlsxwriter")
        try:
            yield writer
            writer.close()
            print(f"⚠ 파일이 사용 중이라 다른 이름으로 저장했습니다: {alt}")
        finally:
            try:
                writer.close()
            except Exception:
                pass

# =========================
# 유틸
# =========================
def to_year(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    dt = pd.to_datetime(s, errors="coerce")   # infer_datetime_format 제거(경고 해소)
    return dt.dt.year

# =========================
# 데이터 로드
# =========================
def load_inputs():
    env = pd.read_excel(ENV_DAILY_PATH)
    fruit = pd.read_excel(FRUIT_TARGET_PATH)

    if COL_DATE not in env.columns:
        raise ValueError(f"기상데이터에 '{COL_DATE}' 컬럼이 없습니다.")
    if COL_REGION not in env.columns:
        raise ValueError(f"기상데이터에 '{COL_REGION}' 컬럼이 없습니다.")
    if COL_REGION not in fruit.columns:
        raise ValueError(f"과실데이터에 '{COL_REGION}' 컬럼이 없습니다.")

    env = env.copy()
    env[COL_DATE] = pd.to_datetime(env[COL_DATE], errors="coerce")  # infer_datetime_format 제거
    if env[COL_DATE].isna().all():
        raise ValueError("기상 일자 파싱 실패")

    env["year"] = env[COL_DATE].dt.year
    env["month"] = env[COL_DATE].dt.month

    env_cols = [c for c in CANDIDATE_COLS if c in env.columns]
    if not env_cols:
        raise ValueError("사용 가능한 기상 변수(CANDIDATE_COLS)가 없습니다.")

    fruit = fruit.copy()
    if "year" not in fruit.columns:
        if "일자" in fruit.columns:
            fruit["year"] = to_year(fruit["일자"])
        elif "계측시간" in fruit.columns:
            fruit["year"] = to_year(fruit["계측시간"])
        else:
            raise ValueError("과실데이터에 '일자'/'계측시간'이 없고 'year'도 없습니다.")
    return env, fruit, env_cols

# =========================
# 월별 집계 → 월별 피처 가로 확장
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

    # month 별로 열 이름: 예) 평균기온_mean_m07 / 강우량_sum_m07
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

# =========================
# 엑셀로 내보내기
# =========================
def export_to_sheets(writer, sheet_prefix, model, eq, merged, corr_s, summary, selected, lasso_coef):
    corr_df = (corr_s.to_frame("corr").reset_index().rename(columns={"index":"feature"})
               .sort_values("corr", key=lambda s: s.abs(), ascending=False))
    params = model.params; pvals = model.pvalues; conf = model.conf_int()
    coef_df = pd.DataFrame({
        "feature": params.index, "coef": params.values,
        "p_value": pvals.values, "ci_low": conf[0].values, "ci_high": conf[1].values
    })
    order = {"const": -1} | {f:i for i,f in enumerate(selected)}
    coef_df["order"] = coef_df["feature"].map(order).fillna(1e6)
    coef_df = coef_df.sort_values(["order","feature"]).drop(columns="order")

    y_true = model.model.endog
    y_pred = model.predict(model.model.exog)
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred})

    summary_df = pd.DataFrame({
        "metric": ["n_samples","r2_in_sample","rmse_in_sample","lasso_alpha","equation"],
        "value": [summary["n_samples"], summary["r2_in_sample"], summary["rmse_in_sample"], summary["lasso_alpha"], eq]
    })
    feats_df = pd.DataFrame({"selected_features": selected})
    lasso_df = (lasso_coef.to_frame("lasso_coef").reset_index().rename(columns={"index":"feature"})
                if isinstance(lasso_coef, pd.Series) else pd.DataFrame(columns=["feature","lasso_coef"]))

    # 시트명 31자 제한 대응
    sn = (sheet_prefix[:28] + "...") if len(sheet_prefix) > 31 else sheet_prefix

    corr_df.to_excel(writer, sheet_name=f"{sn}_01_Corr", index=False)
    summary_df.to_excel(writer, sheet_name=f"{sn}_02_Summary", index=False)
    coef_df.to_excel(writer, sheet_name=f"{sn}_03_Coeff", index=False)
    pred_df.to_excel(writer, sheet_name=f"{sn}_04_Preds", index=False)
    feats_df.to_excel(writer, sheet_name=f"{sn}_05_Feats", index=False)
    lasso_df.to_excel(writer, sheet_name=f"{sn}_06_Lasso", index=False)
    if INCLUDE_MERGED:
        (merged if len(merged) <= 50000 else merged.sample(50000, random_state=1))\
            .to_excel(writer, sheet_name=f"{sn}_07_Merged", index=False)

# =========================
# 타깃별 단일 파이프라인
# =========================
def run_for_target(env_daily, fruit_sub, env_cols, target_col):
    monthly = build_monthly_features(env_daily, env_cols)
    merged = pd.merge(fruit_sub, monthly, on=[COL_REGION, "year"], how="inner").dropna(subset=[target_col])

    # 숫자형 타깃만
    if not pd.api.types.is_numeric_dtype(merged[target_col]):
        merged[target_col] = pd.to_numeric(merged[target_col], errors="coerce")
    merged = merged.dropna(subset=[target_col])

    kept, corr_s = select_by_correlation(merged, target_col, CORR_TOP_K, CORR_MIN_ABS)
    if len(kept) == 0:
        raise RuntimeError(f"{target_col}: 상관 기준을 완화하세요(변수 없음).")
    # === 추가: 환경 변수만 남기기 ===
    env_keywords = ("기온", "습도", "강우", "일사", "결로", "풍속")
    kept = [c for c in kept if any(k in c for k in env_keywords)]

    if len(kept) == 0:
        raise RuntimeError(f"{target_col}: 환경 변수에서 선택된 게 없습니다.")

    X = merged[kept].astype(float).fillna(merged[kept].mean())
    y = merged[target_col].astype(float)

    sel, lasso_coef, alpha = lasso_shrink(X, y, LASSO_ALPHAS, LASSO_FOLDS, RANDOM_STATE)
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
        "lasso_alpha": float(alpha)
    }
    return model, eq, merged, corr_s, summary, features, lasso_coef

# =========================
# 메인
# =========================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    env, fruit, env_cols = load_inputs()

    # 타깃 자동 추출(숫자형 & KEY_COLS 제외)
    num_cols = fruit.select_dtypes(include='number').columns.tolist()
    targets = [c for c in num_cols if c not in KEY_COLS]
    if not targets:
        raise ValueError("과실 타깃 후보(숫자형)가 없습니다.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outfile = OUTDIR / f"통합회귀_환경만_MULTI_{ts}.xlsx"

    # 품종 그룹핑 여부
    cultivars = fruit["품종"].dropna().unique().tolist() if ("품종" in fruit.columns and GROUP_BY_CULTIVAR) else [None]

    with safe_excel_writer(outfile) as writer:
        for cv in cultivars:
            fsub = fruit if cv is None else fruit[fruit["품종"] == cv]
            cv_tag = "" if cv is None else f"_{cv}"

            for tgt in targets:
                try:
                    model, eq, merged, corr_s, summary, features, lasso_coef = run_for_target(env, fsub, env_cols, tgt)
                    sheet_prefix = f"{tgt}{cv_tag}"
                    export_to_sheets(writer, sheet_prefix, model, eq, merged, corr_s, summary, features, lasso_coef)
                    print(f"{sheet_prefix} 완료  R²={summary['r2_in_sample']}, RMSE={summary['rmse_in_sample']}")
                except Exception as e:
                    print(f"[SKIP] {tgt}{cv_tag} 실패:", e)

        # 서식(헤더 볼드/고정)
        wb = writer.book
        bold = wb.add_format({"bold": True})
        for sh in writer.sheets.values():
            sh.set_row(0, None, bold)
            sh.freeze_panes(1, 1)

if __name__ == "__main__":
    main()
