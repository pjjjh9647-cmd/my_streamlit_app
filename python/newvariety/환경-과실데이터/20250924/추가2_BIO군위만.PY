# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")  # GUI 없어도 그림 저장 가능
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 0) 경로/출력폴더 (항상 먼저 정의!)
# =========================================
BASE = Path(r"C:\Users\User\Desktop\mba\환경데이터")
BIO_FILE   = BASE / "_OUT" / "bioclim_19_variables_Gunwi.csv"
ENV_FILE   = BASE / "기상데이터_통합.xlsx"
FRUIT_FILE = BASE / "과실데이터_통합.xlsx"

OUTDIR = BASE / "_ANALYSIS_OUT"
OUTDIR.mkdir(parents=True, exist_ok=True)

def save_csv_safe(df, path):
    try:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(path, encoding="utf-8-sig", index=False)
            print(f"[저장] {path}")
        else:
            print(f"[SKIP] 비어있어 저장하지 않음: {path.name}")
    except Exception as e:
        print(f"[ERROR] {path.name} 저장 실패: {e}")

# =========================================
# 1) 데이터 불러오기 (군위만)
# =========================================
bio = pd.read_csv(BIO_FILE)

env_all   = pd.read_excel(ENV_FILE,   sheet_name=None)
fruit_all = pd.read_excel(FRUIT_FILE, sheet_name=None)

def extract_gunwi(df):
    if "지역명" not in df.columns:
        return None
    mask = df["지역명"].astype(str).str.contains("군위", na=False)
    out = df.loc[mask].copy()
    return out if not out.empty else None

env_frames   = [x for x in (extract_gunwi(df) for df in env_all.values())   if x is not None]
fruit_frames = [x for x in (extract_gunwi(df) for df in fruit_all.values()) if x is not None]

if not env_frames:
    raise RuntimeError("기상데이터에서 군위 행을 찾지 못했습니다.")
if not fruit_frames:
    raise RuntimeError("과실데이터에서 군위 행을 찾지 못했습니다.")

env   = pd.concat(env_frames, ignore_index=True)
fruit = pd.concat(fruit_frames, ignore_index=True)

# 연도 생성
if "일자" in env.columns:
    env["year"] = pd.to_datetime(env["일자"], errors="coerce").dt.year
else:
    raise RuntimeError("기상데이터에 '일자' 컬럼이 필요합니다.")

if "일자" in fruit.columns:
    fruit["year"] = pd.to_datetime(fruit["일자"], errors="coerce").dt.year
elif "연도" in fruit.columns:
    fruit["year"] = pd.to_numeric(fruit["연도"], errors="coerce")
else:
    raise RuntimeError("과실데이터에 '일자' 또는 '연도' 컬럼이 필요합니다.")

# =========================================
# 2) 기상 연도 집계(원자료 요약)
# =========================================
agg = {"평균기온":"mean","최고기온":"mean","최저기온":"mean","강우량":"sum"}
if "일사량" in env.columns: agg["일사량"] = "mean"
if "습도"  in env.columns: agg["습도"]  = "mean"

env_year = (env.dropna(subset=["year"])
              .groupby(["지역명","year"], as_index=False)
              .agg(agg)
              .rename(columns={"평균기온":"tmean","최고기온":"tmax","최저기온":"tmin",
                               "강우량":"prcp","일사량":"rad","습도":"humid"}))

# =========================================
# 3) 과실 연도 집계: 숫자만 평균
# =========================================
drop_keys = {"지역명","일자","연도","year"}
fruit_conv = fruit.copy()
for c in fruit_conv.columns:
    if c not in drop_keys:
        fruit_conv[c] = pd.to_numeric(fruit_conv[c], errors="coerce")

num_cols = [c for c in fruit_conv.columns
            if c not in drop_keys and pd.api.types.is_numeric_dtype(fruit_conv[c])]
if not num_cols:
    raise RuntimeError("과실데이터에서 집계할 숫자 컬럼이 없습니다.")

fruit_year = (fruit_conv.dropna(subset=["year"])
                        .groupby(["지역명","year"], as_index=False)[num_cols]
                        .mean())

# =========================================
# 4) 병합: 연도만으로 (군위만 남겼으므로 지역명 제외)
# =========================================
bio_gw = bio[bio["region"].astype(str).str.contains("군위", na=False)].copy()
bio_gw["year"] = pd.to_numeric(bio_gw["year"], errors="coerce").astype("Int64")
bio_y = bio_gw.groupby("year", as_index=False).mean(numeric_only=True)  # BIO1~BIO19

env_year["year"]   = pd.to_numeric(env_year["year"], errors="coerce").astype("Int64")
fruit_year["year"] = pd.to_numeric(fruit_year["year"], errors="coerce").astype("Int64")

env_y   = env_year.groupby("year", as_index=False).mean(numeric_only=True)    # tmean,tmax,tmin,prcp,rad,humid
fruit_y = fruit_year.groupby("year", as_index=False).mean(numeric_only=True)  # 과실 지표

merged = fruit_y.merge(bio_y, on="year", how="inner").merge(env_y, on="year", how="left")

print("병합 데이터 크기:", merged.shape)
print("BIO year:",   sorted(bio_y["year"].dropna().unique().tolist()))
print("ENV year:",   sorted(env_y["year"].dropna().unique().tolist()))
print("FRUIT year:", sorted(fruit_y["year"].dropna().unique().tolist()))
print("교집합 year:", sorted(set(bio_y["year"].dropna()) & set(env_y["year"].dropna()) & set(fruit_y["year"].dropna())))

if merged.empty:
    # 원인 진단용 파일만 남기고 종료
    save_csv_safe(bio_y,   OUTDIR/"_디버그_BIO_years.csv")
    save_csv_safe(env_y,   OUTDIR/"_디버그_ENV_years.csv")
    save_csv_safe(fruit_y, OUTDIR/"_디버그_FRUIT_years.csv")
    raise SystemExit("병합 결과 0행 → _디버그_* 파일로 연도 분포를 확인하세요.")

# 과실 변수 목록(병합 기준으로 존재하는 것만)
fruit_cols = [c for c in num_cols if c in merged.columns]

# =========================================
# 5) 상관분석: 타깃-기상
# =========================================
corr = merged.corr(numeric_only=True)
climate_cols = [c for c in merged.columns if c.startswith("BIO") or c in ["tmean","tmax","tmin","prcp","rad","humid"]]
fruit_cols   = [c for c in fruit_cols if c in corr.index]
climate_cols = [c for c in climate_cols if c in corr.columns]

selected_pairs = pd.DataFrame()
sel_print = pd.DataFrame()
corr_results = pd.DataFrame()

for THRESH in [0.60, 0.55, 0.50]:
    corr_results = corr.loc[fruit_cols, climate_cols].copy()
    high_mask = corr_results.abs() >= THRESH
    selected_pairs = corr_results.where(high_mask)
    sel_print = selected_pairs.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if not sel_print.empty:
        print(f"\n=== |r| >= {THRESH:.2f} 과실-기상 상관 ===")
        print(sel_print)
        break

# =========================================
# 6) 교차상관 & 공선성(|r|>=0.75)
# =========================================
sel_vars = sel_print.columns.unique().tolist() if not sel_print.empty else []
cross_corr = pd.DataFrame()
if sel_vars:
    cross_corr = corr.loc[sel_vars, sel_vars]
    print("\n=== 선택 변수 교차상관 ===")
    print(cross_corr)

    # 공선성 위험쌍(상삼각)
    mask_upper = np.triu(np.ones_like(cross_corr, dtype=bool), k=1)
    risky_pairs = (pd.DataFrame(cross_corr.values, index=cross_corr.index, columns=cross_corr.columns)
                   .where(mask_upper)).stack().rename("r").reset_index()
    risky_pairs = risky_pairs[risky_pairs["r"].abs() >= 0.75]
    if not risky_pairs.empty:
        print("\n공선성 위험 쌍(|r|>=0.75):")
        print(risky_pairs.sort_values(by="r", key=lambda s: s.abs(), ascending=False))

# =========================================
# 7) VIF
# =========================================
vif_df = pd.DataFrame()
if sel_vars:
    X = merged[sel_vars].dropna()
    if len(X) >= 5 and X.shape[1] >= 1:
        Xc = sm.add_constant(X)
        vif_vals = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
        vif_df = pd.DataFrame({"variable": X.columns, "VIF": vif_vals})
        print("\n=== VIF ===")
        print(vif_df.sort_values("VIF", ascending=False))
    else:
        print("\n표본 부족 또는 변수 수 부족으로 VIF 생략")

# =========================================
# 8) OLS
# =========================================
ols_lines = []
def fit_and_log(y_name, X_cols):
    y = merged[y_name]; X = merged[X_cols]
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 5:
        msg = f"[{y_name}] 표본 부족으로 회귀 생략"
        print(msg); ols_lines.append(msg); return
    model = sm.OLS(df[y_name], sm.add_constant(df[X_cols])).fit()
    line = (f"=== OLS 결과: {y_name} ===\n"
            f"n={len(df)}, R²={model.rsquared:.3f}, adj.R²={model.rsquared_adj:.3f}, "
            f"F={model.fvalue:.3f}, p(F)={model.f_pvalue:.3g}\n"
            f"유의한 변수(p<0.05): {[ix for ix,p in model.pvalues.items() if ix!='const' and p<0.05]}\n")
    print("\n"+line); ols_lines.append(line)

if sel_vars:
    for tgt in fruit_cols:
        fit_and_log(tgt, sel_vars)
else:
    ols_lines.append("선택된 기상 변수가 없어 OLS를 생략합니다.")

# =========================================
# 9) 저장 (존재/비어있음 체크)
# =========================================
# 병합본
save_csv_safe(merged, OUTDIR/"군위_병합데이터.csv")

# 상관 결과
if not corr_results.empty:
    corr_results.to_csv(OUTDIR/"군위_타깃vs기상_상관계수.csv", encoding="utf-8-sig")
    print(f"[저장] {OUTDIR/'군위_타깃vs기상_상관계수.csv'}")
else:
    print("[SKIP] corr_results 비어있음")

if not sel_print.empty:
    sel_print.to_csv(OUTDIR/"군위_고상관_타깃vs기상.csv", encoding="utf-8-sig")
    print(f"[저장] {OUTDIR/'군위_고상관_타깃vs기상.csv'}")

if not cross_corr.empty:
    cross_corr.to_csv(OUTDIR/"군위_선택기상_교차상관.csv", encoding="utf-8-sig")
    print(f"[저장] {OUTDIR/'군위_선택기상_교차상관.csv'}")

if not vif_df.empty:
    vif_df.to_csv(OUTDIR/"군위_VIF.csv", encoding="utf-8-sig", index=False)
    print(f"[저장] {OUTDIR/'군위_VIF.csv'}")

with open(OUTDIR/"군위_OLS_요약.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(ols_lines))
print(f"[저장] {OUTDIR/'군위_OLS_요약.txt'}")

# 히트맵
if sel_vars and len(sel_vars) > 1:
    plt.figure(figsize=(9,7))
    sns.heatmap(merged[sel_vars].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("선택된 기상 변수 교차상관 (군위)")
    plt.tight_layout()
    fig_path = OUTDIR/"군위_선택기상_교차상관_heatmap.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[저장] {fig_path}")

print("\n[완료] 결과 폴더:", OUTDIR)
