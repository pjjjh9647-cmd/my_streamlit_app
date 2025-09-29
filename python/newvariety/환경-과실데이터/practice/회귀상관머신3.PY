# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 경로 설정 =====
ENV_FILE   = r"C:\Users\User\Desktop\환경데이터\기상데이터_통합.xlsx"
FRUIT_FILE = r"C:\Users\User\Desktop\과실데이터\과실데이터_통합.xlsx"
OUTDIR     = Path(r"C:\Users\User\Desktop\분석결과_월단위")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ===== 환경 변수 & 타깃 변수 =====
ENV_COLS = ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
TARGETS  = ["당도","과중","산도","L","a","b"]  # 과실 파일에 있는 타깃만 자동으로 사용됨

# ===== 유틸 =====
def safe_to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_year_in_fruit(fruit: pd.DataFrame) -> pd.DataFrame:
    if "연도" in fruit.columns:
        return fruit
    if "일자" in fruit.columns:
        # 1) 숫자 연도 시도
        yr = pd.to_numeric(fruit["일자"], errors="coerce")
        if yr.notna().any():
            fruit["연도"] = yr.astype("Int64")
            return fruit
        # 2) 날짜 파싱 → 연도
        dt = pd.to_datetime(fruit["일자"], errors="coerce")
        if dt.notna().any():
            fruit["연도"] = dt.dt.year
            return fruit
    raise ValueError("과실데이터에서 연도 정보를 만들 수 없습니다. 연도 컬럼 또는 일자(연도 추출 가능)가 필요합니다.")

# ===== 1) 환경: 일 → 월 집계 =====
def build_env_monthly(env_file: str) -> pd.DataFrame:
    env = pd.read_excel(env_file)
    req_cols = {"지역명","일자"} | set(ENV_COLS)
    miss = req_cols - set(env.columns)
    if miss:
        raise ValueError(f"환경데이터 결측 컬럼: {sorted(miss)}")

    env["일자"] = pd.to_datetime(env["일자"], errors="coerce")
    if env["일자"].isna().all():
        raise ValueError("환경데이터 일자 변환 실패")

    env["연도"] = env["일자"].dt.year
    env["월"]   = env["일자"].dt.month

    agg_map = {
        "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
        "강우량":"sum","일사량":"sum","결로시간":"sum",
        "평균풍속":"mean","최대풍속":"mean"
    }
    env_m = env.groupby(["지역명","연도","월"], as_index=False).agg(agg_map)
    env_m = safe_to_numeric(env_m, ENV_COLS)
    return env_m

# ===== 2) 과실: 연도 정리 =====
def load_fruit(fruit_file: str) -> pd.DataFrame:
    fruit = pd.read_excel(fruit_file)
    need = {"지역명","품종"}
    if not need.issubset(fruit.columns):
        raise ValueError(f"과실데이터 필수 컬럼 누락: {sorted(need)}")
    fruit = ensure_year_in_fruit(fruit)
    # 가능한 타깃만 남김
    actual_targets = [t for t in TARGETS if t in fruit.columns]
    return fruit, actual_targets

# ===== 3) 병합 =====
def build_merged(env_m: pd.DataFrame, fruit: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(env_m, fruit, on=["지역명","연도"], how="left")
    merged = merged[merged["품종"].notna()].copy()
    return merged

# ===== 4A) 월별 상관분석 =====
def run_monthly_correlation(merged: pd.DataFrame, env_cols, targets, outdir: Path):
    corr_rows = []
    corr_dir = outdir / "01_상관분석"
    corr_dir.mkdir(exist_ok=True)

    for cultivar, gdf in merged.groupby("품종"):
        cdir = corr_dir / f"{cultivar}"
        cdir.mkdir(exist_ok=True)
        for m in range(1, 13):
            dfm = gdf[gdf["월"] == m].copy()
            use_cols = [c for c in env_cols if c in dfm.columns] + [t for t in targets if t in dfm.columns]
            sub = dfm[use_cols].dropna()
            if len(sub) < 8:
                continue
            corr = sub.corr(method="pearson")
            # CSV 저장
            csv_path = cdir / f"corr_월{m:02d}.csv"
            corr.to_csv(csv_path, encoding="utf-8-sig", index=True)

            # Heatmap 저장
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=False)
            plt.title(f"[{cultivar}] 월 {m:02d} 상관분석")
            plt.tight_layout()
            plt.savefig(cdir / f"corr_월{m:02d}.png", dpi=200)
            plt.close()

            # 행 형태 요약 축적(환경→타깃만 필터)
            for e in env_cols:
                if e not in corr.index: 
                    continue
                for t in targets:
                    if t in corr.columns:
                        corr_rows.append({
                            "품종": cultivar, "월": m, "X(환경)": e, "Y(타깃)": t, "피어슨r": float(corr.loc[e, t])
                        })

    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(corr_dir/"상관_요약_long.csv", index=False, encoding="utf-8-sig")

# ===== 4B) 월별 OLS 회귀 =====
def run_monthly_ols(merged: pd.DataFrame, env_cols, targets, outdir: Path):
    coef_rows, metric_rows = [], []
    reg_dir = outdir / "02_회귀분석_OLS"
    reg_dir.mkdir(exist_ok=True)

    cultivars = sorted(merged["품종"].dropna().unique().tolist())
    for cultivar in cultivars:
        gdf = merged[merged["품종"] == cultivar].copy()
        for target in targets:
            if target not in gdf.columns:
                continue
            for m in range(1, 13):
                dfm = gdf[gdf["월"] == m]
                cols_exist = [c for c in env_cols if c in dfm.columns]
                if not cols_exist:
                    continue
                data = dfm[cols_exist + [target]].dropna()
                if len(data) < max(10, len(cols_exist) + 5):
                    continue
                X = sm.add_constant(data[cols_exist], has_constant="add")
                y = data[target]
                try:
                    model = sm.OLS(y, X).fit()
                except Exception:
                    continue

                params = model.params.to_dict()
                pvals  = model.pvalues.to_dict()

                # 회귀식 텍스트
                def fmt(v): return f"{v:.6g}"
                eq = f"{target} = {fmt(params.get('const', 0.0))}"
                for name in cols_exist:
                    if name in params:
                        sign = "+" if params[name] >= 0 else ""
                        eq += f" {sign}{fmt(params[name])}*{name}"

                # 콘솔 표시
                print(f"[OLS] 품종={cultivar} 월={m:02d} 타깃={target} N={len(data)} Adj.R2={model.rsquared_adj:.3f}")
                print("식:", eq)

                # 계수/유의성 저장
                for name, val in params.items():
                    coef_rows.append({
                        "품종": cultivar, "월": m, "타깃": target,
                        "변수": name, "계수": float(val),
                        "p_value": float(pvals.get(name, np.nan)),
                        "N": int(len(data))
                    })

                # 성능 저장
                metric_rows.append({
                    "품종": cultivar, "월": m, "타깃": target,
                    "N": int(len(data)), "R2": float(model.rsquared),
                    "Adj_R2": float(model.rsquared_adj),
                    "AIC": float(model.aic), "BIC": float(model.bic)
                })

    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(reg_dir/"회귀계수.csv", index=False, encoding="utf-8-sig")
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(reg_dir/"성능요약.csv", index=False, encoding="utf-8-sig")

# ===== 4C) 월별 랜덤포레스트(머신러닝) =====
def run_monthly_random_forest(merged: pd.DataFrame, env_cols, targets, outdir: Path):
    rf_rows_perf, rf_rows_imp = [], []
    rf_dir = outdir / "03_머신러닝_RF"
    rf_dir.mkdir(exist_ok=True)

    cultivars = sorted(merged["품종"].dropna().unique().tolist())
    for cultivar in cultivars:
        gdf = merged[merged["품종"] == cultivar].copy()
        for target in targets:
            if target not in gdf.columns:
                continue
            for m in range(1, 13):
                dfm = gdf[gdf["월"] == m].copy()
                cols_exist = [c for c in env_cols if c in dfm.columns]
                if not cols_exist:
                    continue
                data = dfm[cols_exist + [target]].dropna()
                if len(data) < max(30, len(cols_exist) * 4):  # RF는 표본을 더 요구
                    continue

                X = data[cols_exist]
                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                rf = RandomForestRegressor(
                    n_estimators=400, random_state=42, n_jobs=-1, max_features="auto"
                )
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)

                r2  = r2_score(y_test, pred)
                rmse = mean_squared_error(y_test, pred, squared=False)

                # 성능 저장
                rf_rows_perf.append({
                    "품종": cultivar, "월": m, "타깃": target,
                    "N_total": int(len(data)),
                    "N_train": int(len(X_train)), "N_test": int(len(X_test)),
                    "R2_test": float(r2), "RMSE_test": float(rmse)
                })

                # 변수 중요도
                for var, imp in zip(cols_exist, rf.feature_importances_):
                    rf_rows_imp.append({
                        "품종": cultivar, "월": m, "타깃": target,
                        "변수": var, "중요도": float(imp)
                    })

    if rf_rows_perf:
        pd.DataFrame(rf_rows_perf).to_csv(rf_dir/"RF_성능요약.csv", index=False, encoding="utf-8-sig")
    if rf_rows_imp:
        imp_df = pd.DataFrame(rf_rows_imp)
        imp_df.to_csv(rf_dir/"RF_변수중요도_raw.csv", index=False, encoding="utf-8-sig")
        # 품종×월×타깃별 상위 중요 변수 5개 요약
        top5 = (imp_df
                .sort_values(["품종","월","타깃","중요도"], ascending=[True, True, True, False])
                .groupby(["품종","월","타깃"])
                .head(5))
        top5.to_csv(rf_dir/"RF_변수중요도_top5.csv", index=False, encoding="utf-8-sig")

# ===== 메인 =====
def main():
    # 데이터 준비
    env_m = build_env_monthly(ENV_FILE)
    fruit, actual_targets = load_fruit(FRUIT_FILE)
    if not actual_targets:
        raise ValueError("과실데이터에서 사용할 타깃 변수가 없습니다(TARGETS 목록 확인).")
    merged = build_merged(env_m, fruit)
    merged.to_excel(OUTDIR/"병합데이터_월단위.xlsx", index=False)

    # 상관분석
    run_monthly_correlation(merged, ENV_COLS, actual_targets, OUTDIR)
    # 회귀분석
    run_monthly_ols(merged, ENV_COLS, actual_targets, OUTDIR)
    # 머신러닝
    run_monthly_random_forest(merged, ENV_COLS, actual_targets, OUTDIR)

    print("\n완료. 결과 폴더:", OUTDIR)

if __name__ == "__main__":
    main()
