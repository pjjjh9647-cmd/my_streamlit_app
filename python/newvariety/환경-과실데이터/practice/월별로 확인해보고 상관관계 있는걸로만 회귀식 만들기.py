# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 사용자 입력 구간 (경로/컬럼)
# =========================
ENV_DAILY_PATH    = r"C:\Users\User\Desktop\환경데이터\기상데이터_통합.xlsx"    # 일별 기상 데이터
FRUIT_TARGET_PATH = r"C:\Users\User\Desktop\과실데이터\과실데이터_통합.xlsx"  # 과실 특성 데이터
OUTDIR            = r"C:\Users\User\Desktop\분석결과"                      # 결과 저장 폴더

# 한글 컬럼명 매핑
COL_DATE   = "일자"     # 기상데이터 날짜
COL_REGION = "지역명"   # 지역 키
# COL_TARGET은 다중 타깃 루프에서 동적으로 바뀜

# 기상 후보 변수(파일에 존재하는 것만 자동 사용)
CANDIDATE_COLS = [
    "평균기온", "최고기온", "최저기온",
    "습도", "강우량", "일사량",
    "결로시간", "평균풍속", "최대풍속"
]

# 과실데이터에서 타깃이 아닌 키 컬럼
KEY_COLS = {"품종", "일자", "지역명", "year"}

# 상관/라쏘/VIF 파라미터
CORR_TOP_K    = 25
CORR_MIN_ABS  = 0.20
VIF_THRESHOLD = 10.0
LASSO_ALPHAS  = np.logspace(-3, 1, 60)
LASSO_FOLDS   = 5
RANDOM_STATE  = 42

INCLUDE_MERGED = True  # 원본 병합 데이터 시트 포함 여부

# =========================
# 데이터 로드 및 전처리
# =========================
def load_inputs():
    env_daily = pd.read_excel(ENV_DAILY_PATH)
    fruit = pd.read_excel(FRUIT_TARGET_PATH)

    if COL_DATE not in env_daily.columns:
        raise ValueError(f"기상데이터에 {COL_DATE} 컬럼이 없습니다")
    if COL_REGION not in env_daily.columns:
        raise ValueError(f"기상데이터에 {COL_REGION} 컬럼이 없습니다")
    if COL_REGION not in fruit.columns:
        raise ValueError(f"과실데이터에 {COL_REGION} 컬럼이 없습니다")

    # 기상 일자 처리
    env_daily = env_daily.copy()
    env_daily[COL_DATE] = pd.to_datetime(env_daily[COL_DATE])
    env_daily["year"] = env_daily[COL_DATE].dt.year
    env_daily["month"] = env_daily[COL_DATE].dt.month

    # 사용 가능한 기상 컬럼만 필터
    env_cols = [c for c in CANDIDATE_COLS if c in env_daily.columns]
    if not env_cols:
        raise ValueError("사용 가능한 기상 변수(CANDIDATE_COLS)가 없습니다")

    # 과실 year 처리: '일자'가 연도 또는 날짜 모두 대응
    fruit = fruit.copy()
    if "year" not in fruit.columns:
        if "일자" in fruit.columns:
            s = fruit["일자"]
            if pd.api.types.is_numeric_dtype(s) and s.ge(1900).all() and s.lt(2100).all():
                fruit["year"] = s.astype(int)
            else:
                fruit["year"] = pd.to_datetime(s).dt.year
        else:
            raise ValueError("과실데이터에 '일자' 또는 'year' 컬럼이 필요합니다")

    return env_daily, fruit, env_cols

# =========================
# 월별 피처 생성
# =========================
def build_monthly_features(env_daily, env_cols, region_col=COL_REGION, date_col=COL_DATE):
    df = env_daily[[region_col, "year", "month"] + env_cols].copy()

    mean_like = [c for c in env_cols if re.search("기온|습도|풍속", c)]
    sum_like  = [c for c in env_cols if re.search("일사|강우|강수|복사|rad|precip", c, re.I)]
    remainder = [c for c in env_cols if c not in mean_like + sum_like]
    mean_like += remainder

    agg_map = {c: "mean" for c in mean_like}
    agg_map.update({c: "sum" for c in sum_like})

    g = df.groupby([region_col, "year", "month"]).agg(agg_map).reset_index()

    wide_list = []
    for m in range(1, 13):
        sub = g[g["month"] == m].drop(columns=["month"])
        sub = sub.rename(columns={c: f"{c}_{agg_map.get(c,'mean')}_m{m}" for c in env_cols})
        wide_list.append(sub)
    wide = wide_list[0]
    for t in wide_list[1:]:
        wide = pd.merge(wide, t, on=[region_col, "year"], how="outer")
    return wide

# =========================
# 변수 선택 도구
# =========================
def select_by_correlation(df_merged, target_col, top_k=CORR_TOP_K, min_abs_corr=CORR_MIN_ABS):
    num_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in ["year"] and c != target_col]
    corrs = df_merged[feats + [target_col]].corr(numeric_only=True)[target_col].drop(target_col)
    corrs = corrs.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    kept = corrs[abs(corrs) >= min_abs_corr].index.tolist()
    if len(kept) > top_k:
        kept = kept[:top_k]
    return kept, corrs

def lasso_shrink(X, y, alphas=LASSO_ALPHAS, n_splits=LASSO_FOLDS, random_state=RANDOM_STATE):
    Xs = (X - X.mean()) / X.std(ddof=0)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    lasso = LassoCV(alphas=alphas, cv=cv, random_state=random_state, max_iter=20000).fit(Xs, y)
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected = coef[coef.abs() > 1e-8].index.tolist()
    return selected, coef, float(lasso.alpha_)

def vif_prune(X, thresh=VIF_THRESHOLD):
    feats = list(X.columns)
    dropped = True
    while dropped and len(feats) > 1:
        dropped = False
        Xc = sm.add_constant(X[feats])
        vifs = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
        vif_series = pd.Series(vifs, index=["const"] + feats).drop("const")
        worst = vif_series.idxmax()
        if vif_series.max() > thresh:
            feats.remove(worst)
            dropped = True
    return feats

def finalize_ols(X, y):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return model

def model_equation_string(model, feature_names, target_name):
    params = model.params
    terms = [f"{params['const']:.6g}"]
    for f in feature_names:
        b = params.get(f, 0.0)
        terms.append(f"{b:+.6g}·{f}")
    return f"{target_name} = " + " ".join(terms)

# =========================
# 엑셀에 타깃별로 시트 추가 저장
# =========================
def export_results_to_excel_append(writer, sheet_prefix, model, equation,
                                   merged_df, corr_series, summary,
                                   feature_order=None, lasso_coef=None,
                                   include_merged=True):
    corr_df = (corr_series.to_frame("corr_with_target").reset_index()
               .rename(columns={"index":"feature"})
               .sort_values("corr_with_target", key=lambda s: s.abs(), ascending=False))

    params = model.params; pvals = model.pvalues; conf = model.conf_int()
    coef_df = pd.DataFrame({"feature": params.index, "coef": params.values,
                            "p_value": pvals.values, "ci_low": conf[0].values, "ci_high": conf[1].values})
    if feature_order:
        order = {"const": -1}; order.update({f:i for i,f in enumerate(feature_order)})
        coef_df["order"] = coef_df["feature"].map(order).fillna(1e6)
        coef_df = coef_df.sort_values(["order","feature"]).drop(columns="order")
    else:
        coef_df = coef_df.sort_values(by="feature")

    y_true = model.model.endog
    y_pred = model.predict(model.model.exog)
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred})

    summary_df = pd.DataFrame({
        "metric": ["n_samples","r2_in_sample","rmse_in_sample","lasso_alpha","equation"],
        "value": [summary.get("n_samples"), summary.get("r2_in_sample"),
                  summary.get("rmse_in_sample"), summary.get("lasso_alpha"), equation]
    })

    features_df = pd.DataFrame({"selected_features": feature_order or []})
    lasso_df = (lasso_coef.to_frame("lasso_coef").reset_index().rename(columns={"index":"feature"})
                if isinstance(lasso_coef, pd.Series) else pd.DataFrame(columns=["feature","lasso_coef"]))

    corr_df.to_excel(writer, sheet_name=f"{sheet_prefix}_01_Corr", index=False)
    summary_df.to_excel(writer, sheet_name=f"{sheet_prefix}_02_Summary", index=False)
    coef_df.to_excel(writer, sheet_name=f"{sheet_prefix}_03_Coeff", index=False)
    pred_df.to_excel(writer, sheet_name=f"{sheet_prefix}_04_Preds", index=False)
    features_df.to_excel(writer, sheet_name=f"{sheet_prefix}_05_Feats", index=False)
    lasso_df.to_excel(writer, sheet_name=f"{sheet_prefix}_06_Lasso", index=False)

    if include_merged and isinstance(merged_df, pd.DataFrame):
        (merged_df if len(merged_df)<=50000 else merged_df.sample(50000, random_state=1))\
            .to_excel(writer, sheet_name=f"{sheet_prefix}_07_Merged", index=False)

# =========================
# 타깃별 모델링 파이프라인
# =========================
def build_single_regression_for_target(env_daily, fruit, env_cols, target_col):
    # 1) 월별 피처 생성 + 병합
    monthly = build_monthly_features(env_daily, env_cols, region_col=COL_REGION, date_col=COL_DATE)
    merged = pd.merge(fruit, monthly, on=[COL_REGION, "year"], how="inner").dropna(subset=[target_col])

    # 2) 상관 기반 1차 후보
    kept, corr_series = select_by_correlation(
        merged, target_col=target_col, top_k=CORR_TOP_K, min_abs_corr=CORR_MIN_ABS
    )
    if len(kept) == 0:
        raise RuntimeError(f"{target_col}: 상관 기준을 완화하세요. 후보 변수가 없습니다")

    # (선택) 환경 변수만 남기고 싶으면 여기에서 필터링
    # env_keywords = ("기온", "습도", "강우", "일사", "결로시간", "풍속")
    # kept = [c for c in kept if any(k in c for k in env_keywords)]

    # 3) 결측치 대체(평균으로) 적용 위치: 여기
    X = merged[kept].astype(float).fillna(merged[kept].mean())
    y = merged[target_col].astype(float)

    # 4) 라쏘 → VIF → 최종 OLS
    lasso_feats, lasso_coef, alpha = lasso_shrink(
        X, y, alphas=LASSO_ALPHAS, n_splits=LASSO_FOLDS, random_state=RANDOM_STATE
    )
    if len(lasso_feats) == 0:
        lasso_feats = kept[:min(5, len(kept))]

    vif_feats = vif_prune(X[lasso_feats], thresh=VIF_THRESHOLD)
    final_feats = vif_feats if len(vif_feats) > 0 else lasso_feats

    model = finalize_ols(X[final_feats], y)
    equation = model_equation_string(model, final_feats, target_name=target_col)

    # 5) 성능
    y_hat = model.predict(sm.add_constant(X[final_feats]))
    r2 = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))

    summary = {
        "n_samples": int(len(merged)),
        "selected_features": final_feats,
        "r2_in_sample": round(float(r2), 3),
        "rmse_in_sample": round(float(rmse), 3),
        "lasso_alpha": float(alpha),
    }
    return model, equation, merged, corr_series, summary, lasso_coef


# =========================
# 메인: 한 파일에 타깃별 시트 저장
# =========================
def main():
    env_daily, fruit, env_cols = load_inputs()

    # 타깃 후보 자동 추출: 숫자형 컬럼 중 키 제외
    numeric_cols = fruit.select_dtypes(include='number').columns.tolist()
    target_candidates = [c for c in numeric_cols if c not in KEY_COLS]
    # 필요 시 수동 지정
    # target_candidates = ["당도", "산도", "경도"]

    os.makedirs(OUTDIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outfile = os.path.join(OUTDIR, f"통합회귀_결과_MULTI_{ts}.xlsx")

    with pd.ExcelWriter(outfile, engine="xlsxwriter") as writer:
        for tgt in target_candidates:
            try:
                model, equation, merged_df, corr_series, summary, lasso_coef = \
                    build_single_regression_for_target(env_daily, fruit, env_cols, tgt)
                export_results_to_excel_append(
                    writer, sheet_prefix=tgt,
                    model=model, equation=equation, merged_df=merged_df,
                    corr_series=corr_series, summary=summary,
                    feature_order=summary.get("selected_features"),
                    lasso_coef=lasso_coef, include_merged=INCLUDE_MERGED
                )
                print(f"{tgt} 완료  R²={summary['r2_in_sample']}, RMSE={summary['rmse_in_sample']}")
            except Exception as e:
                # 실패 타깃은 로그만 남기고 계속 진행
                print(f"{tgt} 처리 실패:", e)

        # 헤더 볼드/고정
        wb = writer.book
        bold = wb.add_format({"bold": True})
        for sh in writer.sheets.values():
            sh.set_row(0, None, bold)
            sh.freeze_panes(1, 1)

    print("저장 완료:", outfile)

if __name__ == "__main__":
    main()
