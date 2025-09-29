# -*- coding: utf-8 -*-
r"""
군위(군위군)만 사용하여
- BIO19, 기상 통합, 과실 통합을 연도 기준으로 병합
- 타깃(과실) vs 기상·BIO 상관, 고상관 변수 선별(임계 0.60→0.55→0.50)
- 교차상관(공선성 확인, |r|>=0.75), VIF
- 단순 OLS(선택 변수로 각 타깃 예측)
- 결과를 _ANALYSIS_OUT 폴더에 저장

실행:
  & "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" "C:\newvariety\run_gunwi_analysis.py"
필요 패키지(없으면 설치):
  pip install pandas numpy statsmodels matplotlib seaborn openpyxl
"""
import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 0) 경로 / 출력 폴더
# ---------------------------------------------------------
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
            print(f"[SKIP] 비어있어 저장 안함: {path.name}")
    except Exception as e:
        print(f"[ERROR] {path.name} 저장 실패: {e}")

# ---------------------------------------------------------
# 날짜/연도 파싱 유틸 (과실 쪽 견고화)
# ---------------------------------------------------------
def to_year_series(s: pd.Series) -> pd.Series:
    """여러 형태(문자 YYYY, YYYY-MM-DD, YYYYMMDD 정수/문자, 엑셀 직렬일)를 연도로 변환"""
    s2 = s.copy()

    # 1) 이미 4자리 연도만 들어있다면
    m_year = s2.astype(str).str.fullmatch(r"\s*(19|20)\d{2}\s*")
    if m_year.fillna(False).all():
        return pd.to_numeric(s2, errors="coerce")

    # 2) YYYYMMDD 정수/문자 (예: 20190930)
    def looks_like_yyyymmdd(x):
        try:
            xs = str(int(float(x))).strip()
        except Exception:
            xs = str(x)
        return bool(re.fullmatch(r"(19|20)\d{6}", xs))
    mask_ymd = s2.apply(looks_like_yyyymmdd)
    if mask_ymd.any():
        tmp = s2[mask_ymd].apply(lambda x: str(int(float(x))).strip())
        y = pd.to_datetime(tmp, format="%Y%m%d", errors="coerce").dt.year
        s2.loc[mask_ymd] = y

    # 3) 엑셀 직렬일(1899-12-30 기준)로 보이는 값 (대략 20000~50000 범위)
    def looks_like_excel_serial(x):
        try:
            v = float(x)
            return 10000 <= v <= 60000
        except Exception:
            return False
    mask_serial = s2.apply(looks_like_excel_serial)
    if mask_serial.any():
        y = pd.to_datetime(s2[mask_serial].astype(float), unit="D", origin="1899-12-30", errors="coerce").dt.year
        s2.loc[mask_serial] = y

    # 4) 일반 문자열 날짜
    s2 = pd.to_datetime(s2, errors="coerce").dt.year

    return pd.to_numeric(s2, errors="coerce").astype("Int64")

def extract_gunwi(df):
    if "지역명" not in df.columns:
        return None
    mask = df["지역명"].astype(str).str.contains("군위", na=False)
    out = df.loc[mask].copy()
    return out if not out.empty else None

# ---------------------------------------------------------
# 1) 데이터 로딩
# ---------------------------------------------------------
print("\n[1] 파일 로딩...")
bio = pd.read_csv(BIO_FILE)
env_all   = pd.read_excel(ENV_FILE,   sheet_name=None)
fruit_all = pd.read_excel(FRUIT_FILE, sheet_name=None)

env_frames   = [x for x in (extract_gunwi(df) for df in env_all.values())   if x is not None]
fruit_frames = [x for x in (extract_gunwi(df) for df in fruit_all.values()) if x is not None]

if not env_frames:
    raise SystemExit("기상데이터에서 '군위' 행을 찾지 못했습니다.")
if not fruit_frames:
    raise SystemExit("과실데이터에서 '군위' 행을 찾지 못했습니다.")

env   = pd.concat(env_frames, ignore_index=True)
fruit = pd.concat(fruit_frames, ignore_index=True)

# ---------------------------------------------------------
# 2) 연도 생성 및 집계
# ---------------------------------------------------------
print("[2] 연도 생성 및 집계...")

# 기상 연도
if "일자" not in env.columns:
    raise SystemExit("기상데이터에 '일자' 컬럼이 필요합니다.")
env["year"] = pd.to_datetime(env["일자"], errors="coerce").dt.year

agg = {"평균기온":"mean","최고기온":"mean","최저기온":"mean","강우량":"sum"}
if "일사량" in env.columns: agg["일사량"] = "mean"
if "습도"  in env.columns: agg["습도"]  = "mean"

env_year = (env.dropna(subset=["year"])
              .groupby(["지역명","year"], as_index=False)
              .agg(agg)
              .rename(columns={
                  "평균기온":"tmean","최고기온":"tmax","최저기온":"tmin",
                  "강우량":"prcp","일사량":"rad","습도":"humid"}))

# 과실 연도: '일자' 열이 이미 연도(정수)라서 그대로 year로 사용
if "일자" in fruit.columns:
    fruit["year"] = pd.to_numeric(fruit["일자"], errors="coerce").astype("Int64")
elif "연도" in fruit.columns:
    fruit["year"] = pd.to_numeric(fruit["연도"], errors="coerce").astype("Int64")
else:
    raise SystemExit("과실데이터에 '일자'(연도) 또는 '연도' 컬럼이 필요합니다.")


# 숫자 컬럼만 평균
drop_keys = {"지역명","일자","연도","year"}
fruit_conv = fruit.copy()
for c in fruit_conv.columns:
    if c not in drop_keys:
        fruit_conv[c] = pd.to_numeric(fruit_conv[c], errors="coerce")

num_cols = [c for c in fruit_conv.columns if c not in drop_keys and pd.api.types.is_numeric_dtype(fruit_conv[c])]
if not num_cols:
    raise SystemExit("과실데이터에서 집계할 숫자 컬럼이 없습니다. 원본의 숫자 형식을 확인하세요.")

fruit_year = (fruit_conv.dropna(subset=["year"])
                         .groupby(["지역명","year"], as_index=False)[num_cols]
                         .mean())

# ---------------------------------------------------------
# 3) 병합: 연도 단위로
# ---------------------------------------------------------
print("[3] 병합 준비...")
bio_gw = bio[bio["region"].astype(str).str.contains("군위", na=False)].copy()
bio_gw["year"] = pd.to_numeric(bio_gw["year"], errors="coerce").astype("Int64")
bio_y = bio_gw.groupby("year", as_index=False).mean(numeric_only=True)

env_year["year"]   = pd.to_numeric(env_year["year"], errors="coerce").astype("Int64")
fruit_year["year"] = pd.to_numeric(fruit_year["year"], errors="coerce").astype("Int64")

env_y   = env_year.groupby("year", as_index=False).mean(numeric_only=True)
fruit_y = fruit_year.groupby("year", as_index=False).mean(numeric_only=True)

# 디버그 저장(연도 확인)
save_csv_safe(bio_y,   OUTDIR/"_디버그_BIO_years.csv")
save_csv_safe(env_y,   OUTDIR/"_디버그_ENV_years.csv")
save_csv_safe(fruit_y, OUTDIR/"_디버그_FRUIT_years.csv")

print("BIO year:",   sorted(map(int, pd.Series(bio_y["year"]).dropna().unique())))
print("ENV year:",   sorted(map(int, pd.Series(env_y["year"]).dropna().unique())))
print("FRUIT year:", sorted(map(int, pd.Series(fruit_y["year"]).dropna().unique())))

merged = fruit_y.merge(bio_y, on="year", how="inner").merge(env_y, on="year", how="left")
print("병합 데이터 크기:", merged.shape)

if merged.empty:
    print("병합 결과 0행 → _디버그_*years.csv 로 연도 교집합 확인 필요")
    raise SystemExit()

# 병합본 저장
save_csv_safe(merged, OUTDIR/"군위_병합데이터.csv")

# ---------------------------------------------------------
# 4) 상관분석 (타깃 vs 기상·BIO)
# ---------------------------------------------------------
print("[4] 상관/고상관 변수 선별...")
corr = merged.corr(numeric_only=True)

# 타깃(과실) 후보: 병합된 컬럼 중 원래 과실 수치 컬럼과 교집합
fruit_cols = [c for c in num_cols if c in merged.columns]
climate_cols = [c for c in merged.columns if c.startswith("BIO") or c in ["tmean","tmax","tmin","prcp","rad","humid"]]

fruit_cols   = [c for c in fruit_cols   if c in corr.index]
climate_cols = [c for c in climate_cols if c in corr.columns]

selected_pairs = pd.DataFrame()
sel_print = pd.DataFrame()
corr_results = pd.DataFrame()

for THRESH in [0.60, 0.55, 0.50]:
    if not fruit_cols or not climate_cols:
        break
    corr_results = corr.loc[fruit_cols, climate_cols].copy()
    high_mask = corr_results.abs() >= THRESH
    selected_pairs = corr_results.where(high_mask)
    sel_print = selected_pairs.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if not sel_print.empty:
        print(f"  - 고상관 발견(|r| >= {THRESH:.2f})")
        break

if not corr_results.empty:
    corr_results.to_csv(OUTDIR/"군위_타깃vs기상_상관계수.csv", encoding="utf-8-sig")
    print(f"[저장] {OUTDIR/'군위_타깃vs기상_상관계수.csv'}")
if not sel_print.empty:
    sel_print.to_csv(OUTDIR/"군위_고상관_타깃vs기상.csv", encoding="utf-8-sig")
    print(f"[저장] {OUTDIR/'군위_고상관_타깃vs기상.csv'}")
elif not corr_results.empty:
    print("  - 기준 0.50까지 낮춰도 고상관 타깃-기상 쌍이 없습니다.")

# ---------------------------------------------------------
# 5) 교차상관(공선성) & VIF
# ---------------------------------------------------------
print("[5] 교차상관/공선성/VIF...")
sel_vars = sel_print.columns.unique().tolist() if not sel_print.empty else []
if sel_vars:
    cross_corr = corr.loc[sel_vars, sel_vars]
    cross_corr.to_csv(OUTDIR/"군위_선택기상_교차상관.csv", encoding="utf-8-sig")
    print(f"[저장] {OUTDIR/'군위_선택기상_교차상관.csv'}")

    # 공선성 위험쌍(|r|>=0.75)
    mask_upper = np.triu(np.ones_like(cross_corr, dtype=bool), k=1)
    risky_pairs = (pd.DataFrame(cross_corr.values, index=cross_corr.index, columns=cross_corr.columns)
                   .where(mask_upper)).stack().rename("r").reset_index()
    risky_pairs = risky_pairs[risky_pairs["r"].abs() >= 0.75]
    if not risky_pairs.empty:
        risky_pairs.sort_values(by="r", key=lambda s: s.abs(), ascending=False).to_csv(
            OUTDIR/"군위_공선성위험쌍_r075이상.csv", encoding="utf-8-sig", index=False
        )
        print(f"[저장] {OUTDIR/'군위_공선성위험쌍_r075이상.csv'}")

# VIF
if sel_vars:
    X = merged[sel_vars].dropna()
    if len(X) >= 5 and X.shape[1] >= 1:
        Xc = sm.add_constant(X)
        vif_vals = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
        pd.DataFrame({"variable": X.columns, "VIF": vif_vals}).sort_values(
            "VIF", ascending=False
        ).to_csv(OUTDIR/"군위_VIF.csv", encoding="utf-8-sig", index=False)
        print(f"[저장] {OUTDIR/'군위_VIF.csv'}")
    else:
        print("  - 표본 부족 또는 변수 수 부족으로 VIF 생략")

# ---------------------------------------------------------
# 6) OLS (선택 변수로 각 타깃 예측)
# ---------------------------------------------------------
print("[6] OLS 회귀...")
ols_lines = []
def fit_and_log(y_name, X_cols):
    y = merged[y_name]; X = merged[X_cols]
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 5:
        msg = f"[{y_name}] 표본 부족으로 회귀 생략 (n={len(df)})"
        print("  -", msg); ols_lines.append(msg); return
    model = sm.OLS(df[y_name], sm.add_constant(df[X_cols])).fit()
    line = (f"=== OLS 결과: {y_name} ===\n"
            f"n={len(df)}, R²={model.rsquared:.3f}, adj.R²={model.rsquared_adj:.3f}, "
            f"F={model.fvalue:.3f}, p(F)={model.f_pvalue:.3g}\n"
            f"유의한 변수(p<0.05): {[ix for ix,p in model.pvalues.items() if ix!='const' and p<0.05]}\n")
    print(line); ols_lines.append(line)

if sel_vars:
    for tgt in fruit_cols:
        fit_and_log(tgt, sel_vars)
else:
    ols_lines.append("선택된 기상 변수가 없어 OLS를 생략합니다.")
with open(OUTDIR/"군위_OLS_요약.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(ols_lines))
print(f"[저장] {OUTDIR/'군위_OLS_요약.txt'}")

# ---------------------------------------------------------
# 7) 히트맵 저장 (선택 변수 교차상관)
# ---------------------------------------------------------
if sel_vars and len(sel_vars) > 1:
    plt.figure(figsize=(9,7))
    sns.heatmap(merged[sel_vars].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("선택된 기상 변수 교차상관 (군위)")
    plt.tight_layout()
    fig_path = OUTDIR/"군위_선택기상_교차상관_heatmap.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[저장] {fig_path}")

# ---------------------------------------------------------
# 8) 완료 안내 + 생성물 나열
# ---------------------------------------------------------
print("\n[완료] 결과 폴더:", OUTDIR)
for p in sorted(OUTDIR.glob("*")):
    print(" -", p.name)
