import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지




# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# ===== 경로 설정 =====
ENV_FILE   = r"C:\Users\User\Desktop\환경데이터\기상데이터_통합.xlsx"
FRUIT_FILE = r"C:\Users\User\Desktop\과실데이터\과실데이터_통합.xlsx"
OUTDIR     = Path(r"C:\Users\User\Desktop\분석결과_월단위")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ===== 환경 변수 & 타깃 후보 =====
ENV_COLS = ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
TARGET_CANDIDATES  = ["당도","과중","산도","L","a","b"]

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
        yr = pd.to_numeric(fruit["일자"], errors="coerce")
        if yr.notna().any():
            fruit["연도"] = yr.astype("Int64")
            return fruit
        dt = pd.to_datetime(fruit["일자"], errors="coerce")
        if dt.notna().any():
            fruit["연도"] = dt.dt.year
            return fruit
    raise ValueError("과실데이터에서 연도 정보를 만들 수 없습니다.")

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

# ===== 2) 과실: 연도 정리 & 타깃 자동 탐지 =====
def load_fruit(fruit_file: str):
    fruit = pd.read_excel(fruit_file)
    need = {"지역명","품종"}
    if not need.issubset(fruit.columns):
        raise ValueError(f"과실데이터 필수 컬럼 누락: {sorted(need)}")
    fruit = ensure_year_in_fruit(fruit)

    targets = [t for t in TARGET_CANDIDATES if t in fruit.columns]
    if not targets:
        exclude = {"지역명","품종","연도","일자"}
        num_cols = [c for c in fruit.columns if c not in exclude and pd.api.types.is_numeric_dtype(fruit[c])]
        if not num_cols:
            raise ValueError("사용 가능한 타깃 변수가 없습니다.")
        targets = num_cols
    return fruit, targets

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
            corr.to_csv(cdir / f"corr_월{m:02d}.csv", encoding="utf-8-sig")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(f"[{cultivar}] 월 {m:02d} 상관분석")
            plt.tight_layout()
            plt.savefig(cdir / f"corr_월{m:02d}.png", dpi=200)
            plt.close()
            for e in env_cols:
                if e not in corr.index: continue
                for t in targets:
                    if t in corr.columns:
                        corr_rows.append({"품종":cultivar,"월":m,"X(환경)":e,"Y(타깃)":t,"피어슨r":float(corr.loc[e,t])})
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(corr_dir/"상관_요약_long.csv", index=False, encoding="utf-8-sig")

# ===== 4B) 월별 OLS 회귀 =====
def run_monthly_ols(merged: pd.DataFrame, env_cols, targets, outdir: Path, verbose_preview: bool = True):
    """
    품종 × 월 × 타깃별 OLS 회귀 실행
    - 회귀계수 + p값: 02_회귀분석_OLS/회귀계수.csv
    - 성능요약(R2 등): 02_회귀분석_OLS/성능요약.csv
    - 회귀식 문자열: 02_회귀분석_OLS/회귀식.csv, 회귀식.txt(사람이 읽기 좋은 버전)
    """
    reg_dir = outdir / "02_회귀분석_OLS"
    reg_dir.mkdir(exist_ok=True)

    coef_rows, metric_rows, eq_rows, eq_lines = [], [], [], []

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

                # ---- 회귀식 문자열 만들기 ----
                def fmt(v): return f"{v:.6g}"
                terms = [f"{('+' if params[k] >= 0 else '')}{fmt(params[k])}*{k}"
                         for k in cols_exist if k in params]
                equation = f"{target} = {fmt(params.get('const', 0.0))} " + " ".join(terms)

                # (옵션) 콘솔 프리뷰
                if verbose_preview:
                    print(f"[OLS] 품종={cultivar} 월={m:02d} 타깃={target} N={len(data)} Adj.R2={model.rsquared_adj:.3f}")
                    print("식:", equation)

                # 저장용 row들
                for name, val in params.items():
                    coef_rows.append({
                        "품종": cultivar, "월": m, "타깃": target,
                        "변수": name, "계수": float(val),
                        "p_value": float(pvals.get(name, np.nan)),
                        "N": int(len(data))
                    })

                metric_rows.append({
                    "품종": cultivar, "월": m, "타깃": target,
                    "N": int(len(data)),
                    "R2": float(model.rsquared),
                    "Adj_R2": float(model.rsquared_adj),
                    "AIC": float(model.aic),
                    "BIC": float(model.bic)
                })

                eq_rows.append({
                    "품종": cultivar, "월": m, "타깃": target,
                    "회귀식": equation,
                    "N": int(len(data)),
                    "Adj_R2": float(model.rsquared_adj)
                })
                eq_lines.append(f"[{cultivar}] 월 {m:02d} / {target} | Adj.R2={model.rsquared_adj:.3f} | {equation}")

    # ---- 파일 저장 ----
    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(reg_dir / "회귀계수.csv", index=False, encoding="utf-8-sig")
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(reg_dir / "성능요약.csv", index=False, encoding="utf-8-sig")
    if eq_rows:
        pd.DataFrame(eq_rows).to_csv(reg_dir / "회귀식.csv", index=False, encoding="utf-8-sig")
        with open(reg_dir / "회귀식.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(eq_lines))

# ===== 4C) 월별 랜덤포레스트 =====
def run_monthly_random_forest(merged, env_cols, targets, outdir: Path):
    rf_rows_perf, rf_rows_imp, logs = [], [], []
    rf_dir = outdir / "03_머신러닝_RF"
    rf_dir.mkdir(exist_ok=True)

    cultivars = sorted(merged["품종"].dropna().unique().tolist())
    for cultivar in cultivars:
        gdf = merged[merged["품종"] == cultivar].copy()
        for target in targets:
            if target not in gdf.columns:
                logs.append(f"[SKIP] 품종={cultivar} 타깃={target} 없음"); continue
            for m in range(1,13):
                dfm = gdf[gdf["월"]==m].copy()
                cols_exist=[c for c in env_cols if c in dfm.columns]
                if not cols_exist: continue
                data=dfm[cols_exist+[target]].dropna()
                N=len(data)
                if N < max(12,len(cols_exist)*2): continue

                X=data[cols_exist]; y=data[target]
                scaler=StandardScaler(); X_scaled=scaler.fit_transform(X)

                if N < 60:
                    # 교차검증
                    n_splits=min(5,max(3,N//4))
                    kf=KFold(n_splits=n_splits,shuffle=True,random_state=42)
                    r2s,rmses=[],[]
                    imps_accum=np.zeros(X.shape[1])
                    for tr,te in kf.split(X_scaled):
                        X_tr,X_te=X_scaled[tr],X_scaled[te]; y_tr,y_te=y.iloc[tr],y.iloc[te]
                        rf=RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=-1)
                        rf.fit(X_tr,y_tr); pred=rf.predict(X_te)
                        r2s.append(r2_score(y_te,pred))
                        rmses.append(sqrt(mean_squared_error(y_te,pred)))
                        imps_accum+=rf.feature_importances_
                    rf_rows_perf.append({"품종":cultivar,"월":m,"타깃":target,"N_total":N,"CV_folds":n_splits,"R2_CV_mean":np.mean(r2s),"RMSE_CV_mean":np.mean(rmses)})
                    for var,imp in zip(cols_exist,imps_accum/n_splits):
                        rf_rows_imp.append({"품종":cultivar,"월":m,"타깃":target,"변수":var,"중요도":float(imp)})
                else:
                    # 홀드아웃
                    X_tr,X_te,y_tr,y_te=train_test_split(X_scaled,y,test_size=0.25,random_state=42)
                    rf=RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=-1)
                    rf.fit(X_tr,y_tr); pred=rf.predict(X_te)
                    r2=r2_score(y_te,pred); rmse=sqrt(mean_squared_error(y_te,pred))
                    rf_rows_perf.append({"품종":cultivar,"월":m,"타깃":target,"N_total":N,"N_train":len(X_tr),"N_test":len(X_te),"R2_test":r2,"RMSE_test":rmse})
                    for var,imp in zip(cols_exist,rf.feature_importances_):
                        rf_rows_imp.append({"품종":cultivar,"월":m,"타깃":target,"변수":var,"중요도":float(imp)})

    if rf_rows_perf: pd.DataFrame(rf_rows_perf).to_csv(rf_dir/"RF_성능요약.csv",index=False,encoding="utf-8-sig")
    if rf_rows_imp:
        imp_df=pd.DataFrame(rf_rows_imp)
        imp_df.to_csv(rf_dir/"RF_변수중요도_raw.csv",index=False,encoding="utf-8-sig")
        top5=(imp_df.sort_values(["품종","월","타깃","중요도"],ascending=[True,True,True,False]).groupby(["품종","월","타깃"]).head(5))
        top5.to_csv(rf_dir/"RF_변수중요도_top5.csv",index=False,encoding="utf-8-sig")

def generate_conclusions(outdir: Path):
    corr_file = outdir / "01_상관분석" / "상관_요약_long.csv"
    ols_coef = outdir / "02_회귀분석_OLS" / "회귀계수.csv"
    ols_metric = outdir / "02_회귀분석_OLS" / "성능요약.csv"
    rf_perf = outdir / "03_머신러닝_RF" / "RF_성능요약.csv"
    rf_imp = outdir / "03_머신러닝_RF" / "RF_변수중요도_top5.csv"

    if not (corr_file.exists() and ols_coef.exists() and ols_metric.exists() and rf_perf.exists() and rf_imp.exists()):
        print("⚠ 일부 결과 파일이 없어 결론 요약을 만들 수 없습니다.")
        return

    df_corr = pd.read_csv(corr_file)
    df_ols_m = pd.read_csv(ols_metric)
    df_ols_c = pd.read_csv(ols_coef)
    df_rf_p = pd.read_csv(rf_perf)
    df_rf_i = pd.read_csv(rf_imp)

    conclusions = []

    for (cultivar, month, target), _ in df_ols_m.groupby(["품종","월","타깃"]):
        # --- OLS 정보 ---
        ols_row = df_ols_m[(df_ols_m["품종"]==cultivar)&(df_ols_m["월"]==month)&(df_ols_m["타깃"]==target)]
        if ols_row.empty: continue
        adjr2 = ols_row["Adj_R2"].iloc[0]

        sig_vars = df_ols_c[(df_ols_c["품종"]==cultivar)&(df_ols_c["월"]==month)&(df_ols_c["타깃"]==target)&(df_ols_c["p_value"]<0.05)]
        sig_vars = sig_vars.sort_values("p_value")["변수"].tolist()[:3]

        # --- RF 정보 ---
        rf_row = df_rf_p[(df_rf_p["품종"]==cultivar)&(df_rf_p["월"]==month)&(df_rf_p["타깃"]==target)]
        if rf_row.empty: continue
        rf_r2 = rf_row.filter(like="R2").iloc[0].max()  # R2_test or R2_CV_mean
        rf_top = df_rf_i[(df_rf_i["품종"]==cultivar)&(df_rf_i["월"]==month)&(df_rf_i["타깃"]==target)]
        rf_top_vars = rf_top.sort_values("중요도",ascending=False)["변수"].tolist()[:3]

        # --- Corr 정보 ---
        corr_sub = df_corr[(df_corr["품종"]==cultivar)&(df_corr["월"]==month)&(df_corr["Y(타깃)"]==target)]
        corr_top = (corr_sub.assign(absval=abs(corr_sub["피어슨r"]))
                    .sort_values("absval",ascending=False)
                    .head(3))
        corr_vars = [f"{r['X(환경)']}({r['피어슨r']:.2f})" for _,r in corr_top.iterrows()]

        # --- 문장 생성 ---
        text = f"[{cultivar} / {month}월 / {target}]\n"
        if adjr2>0.4 and rf_r2>0.5:
            text += "환경요인으로 예측력이 비교적 높게 나타났습니다. "
        else:
            text += "예측력이 제한적이었습니다. "

        if sig_vars:
            text += f"OLS에서는 {', '.join(sig_vars)} 요인이 유의미했습니다. "
        if rf_top_vars:
            text += f"랜덤포레스트에서는 {', '.join(rf_top_vars)} 가(이) 주요 변수로 나타났습니다. "
        if corr_vars:
            text += f"상관분석에서도 {', '.join(corr_vars)} 와 같은 경향을 확인했습니다."

        conclusions.append({
            "품종": cultivar, "월": month, "타깃": target,
            "결론": text
        })

    df_out = pd.DataFrame(conclusions)
    df_out.to_excel(outdir/"자동_결론요약.xlsx", index=False)
    with open(outdir/"자동_결론요약.txt","w",encoding="utf-8") as f:
        f.write("\n\n".join(df_out["결론"].tolist()))

    print("자동 결론 요약 저장 완료:", outdir/"자동_결론요약.xlsx")

# ===== 메인 =====
def main():
    env_m=build_env_monthly(ENV_FILE)
    fruit,targets=load_fruit(FRUIT_FILE)
    merged=build_merged(env_m,fruit)
    merged.to_excel(OUTDIR/"병합데이터_월단위.xlsx",index=False)
    run_monthly_correlation(merged,ENV_COLS,targets,OUTDIR)
    run_monthly_ols(merged,ENV_COLS,targets,OUTDIR)
    run_monthly_random_forest(merged,ENV_COLS,targets,OUTDIR)
    print("\n완료. 결과 폴더:",OUTDIR)

if __name__=="__main__":
    main()
    generate_conclusions(OUTDIR)

